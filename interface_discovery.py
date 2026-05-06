#!/usr/bin/env python3
# Copyright (c) 2025 Arista Networks, Inc.
# Use of this source code is governed by the Apache License 2.0
# that can be found in the LICENSE file.
"""Network Interface Discovery Daemon.

Discovers physical network interfaces (both SR-IOV and non-SR-IOV) on the host
and creates/updates NodeInterfaceState custom resources. This provides interface
inventory for CloudVision integration.

For SR-IOV capable interfaces, discovers PF/VF hierarchy.
For regular interfaces, discovers basic interface information.

Runs as a daemonset in hostNetwork mode on every node.
"""

import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from kubernetes import client, config

from constants import (
    CV_INTERFACE_GROUP,
    CV_INTERFACE_VERSION,
    CV_INTERFACE_PLURAL,
    CV_INTERFACE_NAMESPACE,
    DISCOVERY_INTERVAL,
    SYSFS_NET_PATH,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def is_valid_ethernet_mac(mac: str) -> bool:
    """Check if a MAC address is a valid 6-byte Ethernet MAC (AA:BB:CC:DD:EE:FF).

    Rejects InfiniBand 20-byte GIDs and other non-Ethernet address formats.
    """
    parts = mac.split(":")
    if len(parts) != 6:
        return False
    return all(len(p) == 2 and all(c in "0123456789abcdef" for c in p)
               for p in parts)


def get_interface_mac(interface_name: str) -> Optional[str]:
    """Get MAC address for a network interface."""
    try:
        mac_path = SYSFS_NET_PATH / interface_name / "address"
        if mac_path.exists():
            mac = mac_path.read_text().strip().lower()
            if mac and mac != "00:00:00:00:00:00" and is_valid_ethernet_mac(mac):
                return mac
            if mac and not is_valid_ethernet_mac(mac):
                logger.info(
                    "Skipping non-Ethernet address for %s: %s "
                    "(expected 6-byte MAC, got %d-byte address)",
                    interface_name, mac, len(mac.split(":")))
    except Exception as e:
        logger.debug("Failed to read MAC for %s: %s", interface_name, e)
    return None


def get_interface_ip(interface_name: str) -> Optional[str]:
    """Get IPv4 address for a network interface.

    Reads from /sys/class/net/{iface}/address and uses socket ioctl.
    This is more reliable than parsing /proc/net/fib_trie.
    """
    import socket
    import struct
    import fcntl

    try:
        # Read interface operstate to check if it's up
        operstate_path = SYSFS_NET_PATH / interface_name / "operstate"
        if operstate_path.exists():
            operstate = operstate_path.read_text().strip()
            if operstate != "up":
                logger.debug("Interface %s is not up (state: %s)",
                             interface_name, operstate)
                return None

        # Use ioctl to get IP address
        # SIOCGIFADDR = 0x8915 (get interface address)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Pack interface name into struct
            ifreq = struct.pack('256s', interface_name[:15].encode('utf-8'))
            # Get IP address using ioctl
            result = fcntl.ioctl(sock.fileno(), 0x8915, ifreq)
            # Unpack the result to get IP address
            ip_addr = socket.inet_ntoa(result[20:24])

            # Validate it's a real IP (not 127.0.0.1, not 0.0.0.0)
            if ip_addr and ip_addr not in ("127.0.0.1", "0.0.0.0"):
                logger.debug("Found IP %s for interface %s", ip_addr,
                             interface_name)
                return ip_addr
        finally:
            sock.close()

        logger.debug("No IP address found for %s", interface_name)
        return None

    except OSError as e:
        # EADDRNOTAVAIL (99) or other errors mean no IP assigned
        logger.debug("No IP for %s: %s", interface_name, e)
        return None
    except Exception as e:
        logger.debug("Failed to get IP for %s: %s", interface_name, e)
        return None


def get_pci_address(interface_name: str) -> Optional[str]:
    """Get PCI address for a network interface."""
    try:
        device_path = SYSFS_NET_PATH / interface_name / "device"
        if device_path.exists() and device_path.is_symlink():
            # Resolve symlink to get PCI address from path
            real_path = device_path.resolve()
            # PCI address is typically the last component, e.g., 0000:17:00.0
            pci_addr = real_path.name
            if ":" in pci_addr:
                return pci_addr
    except Exception as e:
        logger.debug("Failed to get PCI address for %s: %s", interface_name, e)
    return None


def get_rdma_device(interface_name: str) -> Optional[str]:
    """Get RDMA device name for a network interface.

    RDMA devices are found in /sys/class/infiniband/ or via the device's
    infiniband_verbs or infiniband directory. Common for Mellanox/NVIDIA NICs.
    """
    try:
        device_path = SYSFS_NET_PATH / interface_name / "device"
        if not device_path.exists():
            logger.debug("No device path for %s", interface_name)
            return None

        device_real_path = device_path.resolve()

        # Method 1: Check for infiniband subdirectory (most direct)
        # This contains the actual IB device name
        ib_path = device_real_path / "infiniband"
        if ib_path.exists() and ib_path.is_dir():
            for ib_dev in ib_path.iterdir():
                if ib_dev.is_dir():
                    rdma_name = ib_dev.name
                    logger.debug(
                        "Found RDMA device %s for %s via infiniband dir",
                        rdma_name, interface_name)
                    return rdma_name

        # Method 2: Check for infiniband_verbs subdirectory
        ib_verbs_path = device_real_path / "infiniband_verbs"
        if ib_verbs_path.exists() and ib_verbs_path.is_dir():
            # The verbs device (uverbsN) links to the IB device
            # Check /sys/class/infiniband to find the matching device
            ib_class_path = Path("/sys/class/infiniband")
            if ib_class_path.exists():
                for ib_dev in ib_class_path.iterdir():
                    if ib_dev.is_symlink():
                        ib_dev_real = ib_dev.resolve()
                        # Check if this IB device points to our PCI device
                        if device_real_path == ib_dev_real or device_real_path in ib_dev_real.parents:
                            rdma_name = ib_dev.name
                            logger.debug(
                                "Found RDMA device %s for %s via infiniband_verbs",
                                rdma_name, interface_name)
                            return rdma_name

        # Method 3: Check /sys/class/infiniband directly by PCI address
        pci_addr = get_pci_address(interface_name)
        if pci_addr:
            ib_class_path = Path("/sys/class/infiniband")
            if ib_class_path.exists():
                for ib_dev in ib_class_path.iterdir():
                    if ib_dev.is_symlink():
                        ib_dev_path = ib_dev.resolve()
                        # Check if PCI address matches in the path
                        if pci_addr in str(ib_dev_path):
                            rdma_name = ib_dev.name
                            logger.debug(
                                "Found RDMA device %s for %s via PCI match",
                                rdma_name, interface_name)
                            return rdma_name

        logger.debug("No RDMA device found for %s", interface_name)

    except Exception as e:
        logger.debug("Failed to get RDMA device for %s: %s", interface_name, e)

    return None


def is_physical_interface(interface_name: str) -> bool:
    """Check if interface is a physical network interface.

    Physical interfaces have a 'device' symlink pointing to the PCI device.
    Virtual interfaces (veth, bridges, etc.) do not have this.
    """
    try:
        device_path = SYSFS_NET_PATH / interface_name / "device"
        return device_path.exists() and device_path.is_symlink()
    except Exception:
        return False


def is_sriov_pf(interface_name: str) -> bool:
    """Check if interface is an SR-IOV Physical Function."""
    try:
        sriov_totalvfs_path = SYSFS_NET_PATH / interface_name / "device" / "sriov_totalvfs"
        return sriov_totalvfs_path.exists()
    except Exception:
        return False


def has_vfs_configured(interface_name: str) -> bool:
    """Check if interface has any VFs configured."""
    try:
        sriov_numvfs_path = SYSFS_NET_PATH / interface_name / "device" / "sriov_numvfs"
        if sriov_numvfs_path.exists():
            return int(sriov_numvfs_path.read_text().strip()) > 0
    except Exception:
        pass
    return False


def is_sriov_vf(interface_name: str) -> bool:
    """Check if interface is an SR-IOV Virtual Function.

    VFs have a 'physfn' symlink in their device directory pointing to the parent PF.
    """
    try:
        device_path = SYSFS_NET_PATH / interface_name / "device"
        if not device_path.exists():
            return False

        # Check for physfn symlink (standard SR-IOV VF indicator)
        physfn_path = device_path / "physfn"
        if physfn_path.exists() and physfn_path.is_symlink():
            return True

        # Additional check: VFs typically don't have sriov_totalvfs
        # but their parent PF does. If we have a device but no sriov_totalvfs,
        # check if this device appears as a virtfn under any PF
        sriov_totalvfs_path = device_path / "sriov_totalvfs"
        if sriov_totalvfs_path.exists():
            # Has sriov_totalvfs, so it's a PF, not a VF
            return False

        # Get PCI address of this interface
        pci_addr = get_pci_address(interface_name)
        if not pci_addr:
            return False

        # Check if this PCI address appears as a virtfn under any other interface
        for iface_path in SYSFS_NET_PATH.iterdir():
            if not iface_path.is_dir() or iface_path.name == interface_name:
                continue

            iface_device_path = iface_path / "device"
            if not iface_device_path.exists():
                continue

            # Check if this interface has virtfn* directories
            for vf_dir in iface_device_path.glob("virtfn*"):
                if vf_dir.is_symlink():
                    vf_pci_path = vf_dir.resolve()
                    if vf_pci_path.name == pci_addr:
                        # This interface's PCI address is listed as a VF of another interface
                        logger.debug(
                            "%s is a VF of %s (found via virtfn scan)",
                            interface_name, iface_path.name)
                        return True

        return False
    except Exception as e:
        logger.debug("Error checking if %s is VF: %s", interface_name, e)
        return False


def get_vf_interfaces(pf_name: str, pf_pci: str,
                      vf_cache: 'VFCache') -> List[Dict[str, str]]:
    """Get list of VF interfaces for a given PF.

    Reads VF information from sysfs PCI device tree, which is stable
    regardless of whether VFs are in the host namespace or moved to pods.

    Uses VFCache to remember VF details (name/MAC/IP/RDMA) even when VFs
    are moved to pod namespaces, ensuring stable state reporting.

    Args:
        pf_name: Physical function interface name
        pf_pci: Physical function PCI address
        vf_cache: VFCache instance to store/retrieve VF details
    """
    vfs = []
    if not pf_pci:
        return vfs

    try:
        device_path = SYSFS_NET_PATH / pf_name / "device"
        # Iterate through virtfnX directories (stable, always present)
        for vf_dir in sorted(device_path.glob("virtfn*")):
            vf_pci_path = vf_dir.resolve()
            vf_pci = vf_pci_path.name

            # Try to find the network interface name for this VF
            # Note: net/ directory may be empty if VF is in a pod's namespace
            net_path = vf_pci_path / "net"
            vf_name = None
            vf_mac = None
            vf_ip = None
            vf_rdma = None

            if net_path.exists() and any(net_path.iterdir()):
                # VF is visible in host namespace - get current details
                for vf_iface in net_path.iterdir():
                    vf_name = vf_iface.name
                    vf_mac = get_interface_mac(vf_name)
                    vf_ip = get_interface_ip(vf_name)
                    vf_rdma = get_rdma_device(vf_name)
                    break  # Only one interface per VF

                # Update cache with current details
                if vf_mac:  # Only cache if we got valid data
                    cache_entry = {"pciAddress": vf_pci}
                    if vf_name:
                        cache_entry["name"] = vf_name
                    if vf_mac:
                        cache_entry["mac"] = vf_mac
                    if vf_ip:
                        cache_entry["ip"] = vf_ip
                    if vf_rdma:
                        cache_entry["rdmaDevice"] = vf_rdma
                    vf_cache.update(vf_pci, cache_entry)
            else:
                # VF is in a pod's namespace - use cached details if available
                cached = vf_cache.get(vf_pci)
                if cached:
                    vf_name = cached.get("name")
                    vf_mac = cached.get("mac")
                    vf_ip = cached.get("ip")
                    vf_rdma = cached.get("rdmaDevice")
                    logger.debug(
                        "Using cached details for VF %s (in pod namespace)",
                        vf_pci)
                else:
                    # No cached data and VF not visible - skip this VF
                    # This happens on first boot when VFs are already in pods
                    logger.debug(
                        "Skipping VF %s (in pod namespace, no cached data yet)",
                        vf_pci)
                    continue

            # Build VF info from current or cached data
            # Only add VF if we have at least a MAC address (valid data)
            if vf_mac:
                vf_info = {
                    "pciAddress": vf_pci,
                    "mac": vf_mac,
                }
                if vf_name:
                    vf_info["name"] = vf_name
                if vf_ip:
                    vf_info["ip"] = vf_ip
                if vf_rdma:
                    vf_info["rdmaDevice"] = vf_rdma

                vfs.append(vf_info)
            else:
                # No valid MAC - don't report this VF
                logger.debug("Skipping VF %s (no MAC address available)",
                             vf_pci)

    except Exception as e:
        logger.debug("Failed to enumerate VFs for %s: %s", pf_name, e)

    return vfs


def discover_interfaces(vf_cache: 'VFCache') -> List[Dict]:
    """Discover all physical network interfaces on the host.

    Discovers both SR-IOV capable and regular physical interfaces.
    For SR-IOV PFs, includes VF hierarchy.
    For regular interfaces, includes basic info without VFs.

    Args:
        vf_cache: VFCache instance to maintain stable VF state across namespace moves
    """
    interfaces = []

    try:
        # Iterate through all network interfaces
        for iface_path in sorted(SYSFS_NET_PATH.iterdir()):
            if not iface_path.is_dir():
                continue

            iface_name = iface_path.name

            # Skip loopback and known virtual interfaces
            if iface_name in ("lo", "docker0",
                              "cni0") or iface_name.startswith("veth"):
                continue

            # Check if this is a physical interface (has device symlink)
            if not is_physical_interface(iface_name):
                logger.debug("Skipping %s (not a physical interface)",
                             iface_name)
                continue

            # Skip SR-IOV VFs - they will be discovered as part of their parent PF
            if is_sriov_vf(iface_name):
                logger.debug(
                    "Skipping %s (SR-IOV VF, will be included in parent PF)",
                    iface_name)
                continue

            # Get basic interface information
            iface_mac = get_interface_mac(iface_name)
            if not iface_mac:
                logger.debug("Skipping %s (no MAC address)", iface_name)
                continue

            iface_pci = get_pci_address(iface_name)
            iface_ip = get_interface_ip(iface_name)
            iface_rdma = get_rdma_device(iface_name)

            logger.debug("Interface %s: PCI=%s, IP=%s, RDMA=%s", iface_name,
                         iface_pci, iface_ip, iface_rdma)

            # Check if this is an SR-IOV capable PF
            if is_sriov_pf(iface_name):
                # Discover as SR-IOV PF with VFs
                vfs = get_vf_interfaces(iface_name, iface_pci, vf_cache)

                iface_info = {
                    "name": iface_name,
                    "mac": iface_mac,
                    "pciAddress": iface_pci or "",
                    "Vfs": vfs,
                }
                if iface_ip:
                    iface_info["ip"] = iface_ip
                if iface_rdma:
                    iface_info["rdmaDevice"] = iface_rdma
                    logger.debug("Adding RDMA device %s to interface %s",
                                 iface_rdma, iface_name)

                interfaces.append(iface_info)
                logger.debug(
                    "Discovered SR-IOV PF %s (MAC: %s, RDMA: %s, VFs: %d)",
                    iface_name, iface_mac, iface_rdma or "none", len(vfs))
            else:
                # Discover as regular physical interface (no SR-IOV)
                iface_info = {
                    "name": iface_name,
                    "mac": iface_mac,
                    "pciAddress": iface_pci or "",
                    "Vfs": [],
                }
                if iface_ip:
                    iface_info["ip"] = iface_ip
                if iface_rdma:
                    iface_info["rdmaDevice"] = iface_rdma
                    logger.debug("Adding RDMA device %s to interface %s",
                                 iface_rdma, iface_name)

                interfaces.append(iface_info)
                logger.debug(
                    "Discovered physical interface %s (MAC: %s, IP: %s, RDMA: %s)",
                    iface_name, iface_mac, iface_ip or "none", iface_rdma
                    or "none")

    except Exception as e:
        logger.error("[DISCOVERY] Failed to discover network interfaces: %s",
                     e,
                     exc_info=True)

    return interfaces


class VFCache:
    """Cache for VF details to maintain stable state across namespace moves.

    When VFs are moved to pod namespaces, they disappear from the host's view.
    This cache remembers their name/MAC/IP/RDMA so we can report consistent
    state and avoid unnecessary NodeConfig updates.

    Keyed by PCI address (stable identifier).
    """

    def __init__(self):
        # Map of PCI address -> VF info dict
        self._cache: Dict[str, Dict[str, str]] = {}

    def update(self, pci_address: str, vf_info: Dict[str, str]) -> None:
        """Update cache with VF information.

        Only updates fields that are present in vf_info (doesn't clear existing data).
        """
        if pci_address not in self._cache:
            self._cache[pci_address] = {}

        # Update only the fields that are present
        for key in ["name", "mac", "ip", "rdmaDevice"]:
            if key in vf_info and vf_info[key]:
                self._cache[pci_address][key] = vf_info[key]

    def get(self, pci_address: str) -> Dict[str, str]:
        """Get cached VF information by PCI address.

        Returns empty dict if not in cache.
        """
        return self._cache.get(pci_address, {})

    def get_cached_field(self, pci_address: str, field: str) -> Optional[str]:
        """Get a specific cached field for a VF."""
        return self._cache.get(pci_address, {}).get(field)


class NodeStateCache:
    """Cache for NodeInterfaceState to avoid unnecessary API calls.

    Since this daemonset is the sole writer of the NodeInterfaceState CR,
    we can cache the last known state in memory and only update when changes occur.
    """

    def __init__(self):
        self.interfaces: Optional[List[Dict]] = None
        self.resource_version: Optional[str] = None
        self.initialized: bool = False


def create_or_update_nodestate(api: client.CustomObjectsApi,
                               v1_api: client.CoreV1Api, node_name: str,
                               interfaces: List[Dict],
                               cache: NodeStateCache) -> bool:
    """Create or update NodeInterfaceState CR for this node.

    Uses in-memory cache to avoid unnecessary API calls and comparisons.
    Sets owner reference to the Node object for automatic cleanup on node deletion.

    Returns True if created/updated, False on error.
    """
    # Check cache first - if interfaces haven't changed, skip API call
    if cache.initialized and cache.interfaces == interfaces:
        logger.debug("No changes detected for node %s (cached)", node_name)
        return True

    # Get Node object to set owner reference
    # This ensures the CR is automatically deleted when the node is deleted
    try:
        node = v1_api.read_node(name=node_name)
        owner_references = [{
            "apiVersion": "v1",
            "kind": "Node",
            "name": node_name,
            "uid": node.metadata.uid,
            "controller": False,
            "blockOwnerDeletion": False,
        }]
    except client.exceptions.ApiException as e:
        logger.error(
            "[DISCOVERY] Failed to get Node object for %s: %s. "
            "Creating CR without owner reference.", node_name, e)
        owner_references = []
    except Exception as e:
        logger.error(
            "[DISCOVERY] Unexpected error getting Node object for %s: %s. "
            "Creating CR without owner reference.", node_name, e)
        owner_references = []

    body = {
        "apiVersion": f"{CV_INTERFACE_GROUP}/{CV_INTERFACE_VERSION}",
        "kind": "NodeInterfaceState",
        "metadata": {
            "name": node_name,
            "namespace": CV_INTERFACE_NAMESPACE,
            "labels": {
                "app": "cv-interface-discovery",
            },
        },
        "status": {
            "interfaces": interfaces,
            "syncStatus": "Succeeded",
        },
    }

    # Add owner references if we successfully got the Node object
    if owner_references:
        body["metadata"]["ownerReferences"] = owner_references

    try:
        if not cache.initialized:
            # First run - try to get existing resource to initialize cache
            try:
                existing = api.get_namespaced_custom_object(
                    group=CV_INTERFACE_GROUP,
                    version=CV_INTERFACE_VERSION,
                    namespace=CV_INTERFACE_NAMESPACE,
                    plural=CV_INTERFACE_PLURAL,
                    name=node_name,
                )

                # Initialize cache from existing resource
                cache.resource_version = existing["metadata"][
                    "resourceVersion"]
                cache.interfaces = existing.get("status",
                                                {}).get("interfaces", [])
                cache.initialized = True

                # Check if update is needed
                if cache.interfaces == interfaces:
                    logger.debug("No changes detected for node %s (first run)",
                                 node_name)
                    return True

                # Update needed - preserve resourceVersion
                body["metadata"]["resourceVersion"] = cache.resource_version

                updated = api.replace_namespaced_custom_object_status(
                    group=CV_INTERFACE_GROUP,
                    version=CV_INTERFACE_VERSION,
                    namespace=CV_INTERFACE_NAMESPACE,
                    plural=CV_INTERFACE_PLURAL,
                    name=node_name,
                    body=body,
                )

                # Log what changed
                _log_interface_changes(node_name, cache.interfaces, interfaces)

                # Update cache with new state and resourceVersion
                cache.interfaces = interfaces
                cache.resource_version = updated["metadata"]["resourceVersion"]
                return True

            except client.exceptions.ApiException as e:
                if e.status == 404:
                    # Resource doesn't exist, create it
                    # Note: Create without status first, then update status subresource
                    create_body = {
                        "apiVersion": body["apiVersion"],
                        "kind": body["kind"],
                        "metadata": body["metadata"],
                    }
                    created = api.create_namespaced_custom_object(
                        group=CV_INTERFACE_GROUP,
                        version=CV_INTERFACE_VERSION,
                        namespace=CV_INTERFACE_NAMESPACE,
                        plural=CV_INTERFACE_PLURAL,
                        body=create_body,
                    )

                    # Now update the status subresource
                    body["metadata"]["resourceVersion"] = created["metadata"][
                        "resourceVersion"]
                    updated = api.replace_namespaced_custom_object_status(
                        group=CV_INTERFACE_GROUP,
                        version=CV_INTERFACE_VERSION,
                        namespace=CV_INTERFACE_NAMESPACE,
                        plural=CV_INTERFACE_PLURAL,
                        name=node_name,
                        body=body,
                    )

                    # Log first-time discovery
                    _log_first_discovery(node_name, interfaces)

                    # Initialize cache with updated resource's resourceVersion
                    cache.interfaces = interfaces
                    cache.resource_version = updated["metadata"][
                        "resourceVersion"]
                    cache.initialized = True
                    return True
                else:
                    raise
        else:
            # Cache is initialized and interfaces have changed
            if not cache.resource_version:
                # This shouldn't happen, but if it does, fetch current state
                logger.warning(
                    "Cache missing resourceVersion, fetching current state")
                existing = api.get_namespaced_custom_object(
                    group=CV_INTERFACE_GROUP,
                    version=CV_INTERFACE_VERSION,
                    namespace=CV_INTERFACE_NAMESPACE,
                    plural=CV_INTERFACE_PLURAL,
                    name=node_name,
                )
                cache.resource_version = existing["metadata"][
                    "resourceVersion"]

            body["metadata"]["resourceVersion"] = cache.resource_version

            updated = api.replace_namespaced_custom_object_status(
                group=CV_INTERFACE_GROUP,
                version=CV_INTERFACE_VERSION,
                namespace=CV_INTERFACE_NAMESPACE,
                plural=CV_INTERFACE_PLURAL,
                name=node_name,
                body=body,
            )

            # Log what changed
            _log_interface_changes(node_name, cache.interfaces, interfaces)

            # Update cache with new state and resourceVersion
            cache.interfaces = interfaces
            cache.resource_version = updated["metadata"]["resourceVersion"]
            return True

    except Exception as e:
        logger.error(
            "[DISCOVERY] Failed to create/update NodeInterfaceState for node %s: %s",
            node_name,
            e,
            exc_info=True)
        return False


def _log_first_discovery(node_name: str, interfaces: List[Dict]) -> None:
    """Log first-time interface discovery."""
    sriov_count = sum(1 for iface in interfaces if iface.get("Vfs"))
    regular_count = len(interfaces) - sriov_count

    logger.info(
        "[DISCOVERY] Created NodeInterfaceState for node %s: %d interfaces (%d SR-IOV PFs, %d regular)",
        node_name, len(interfaces), sriov_count, regular_count)

    # Log interface details
    for iface in interfaces:
        vf_count = len(iface.get("Vfs", []))
        rdma = iface.get("rdmaDevice", "")
        if vf_count > 0:
            logger.info(
                "[DISCOVERY]   SR-IOV PF: %s (MAC: %s, RDMA: %s, VFs: %d)",
                iface["name"], iface["mac"], rdma or "none", vf_count)
        else:
            logger.info("[DISCOVERY]   Physical: %s (MAC: %s, RDMA: %s)",
                        iface["name"], iface["mac"], rdma or "none")


def _log_interface_changes(node_name: str, old_interfaces: List[Dict],
                           new_interfaces: List[Dict]) -> None:
    """Log changes between old and new interface states."""
    old_names = {iface["name"] for iface in old_interfaces}
    new_names = {iface["name"] for iface in new_interfaces}

    added = new_names - old_names
    removed = old_names - new_names

    # Build a map for easy comparison
    old_map = {iface["name"]: iface for iface in old_interfaces}
    new_map = {iface["name"]: iface for iface in new_interfaces}

    changes = []

    if added:
        changes.append(f"{len(added)} added")
        for name in added:
            iface = new_map[name]
            rdma = iface.get("rdmaDevice", "")
            logger.info("[DISCOVERY]   Added: %s (MAC: %s, RDMA: %s)", name,
                        iface["mac"], rdma or "none")

    if removed:
        changes.append(f"{len(removed)} removed")
        for name in removed:
            logger.info("[DISCOVERY]   Removed: %s", name)

    # Check for changes in existing interfaces
    modified = []
    for name in old_names & new_names:
        old_iface = old_map[name]
        new_iface = new_map[name]

        # Check for relevant changes (MAC, RDMA, VF count)
        old_rdma = old_iface.get("rdmaDevice", "none")
        new_rdma = new_iface.get("rdmaDevice", "none")

        if (old_iface.get("mac") != new_iface.get("mac")
                or old_rdma != new_rdma or len(old_iface.get(
                    "Vfs", [])) != len(new_iface.get("Vfs", []))):
            modified.append(name)
            old_vfs = len(old_iface.get("Vfs", []))
            new_vfs = len(new_iface.get("Vfs", []))

            logger.info("[DISCOVERY]   Modified: %s (RDMA: %s→%s, VFs: %d→%d)",
                        name, old_rdma, new_rdma, old_vfs, new_vfs)

    if modified:
        changes.append(f"{len(modified)} modified")

    if changes:
        logger.info("[DISCOVERY] Updated NodeInterfaceState for node %s: %s",
                    node_name, ", ".join(changes))
    else:
        logger.info(
            "[DISCOVERY] Updated NodeInterfaceState for node %s: no significant changes",
            node_name)


def main():
    """Main discovery loop."""
    # Get node name from environment or hostname
    node_name = os.environ.get("NODE_NAME")
    if not node_name:
        node_name = socket.gethostname()
        logger.warning("[INIT] NODE_NAME not set, using hostname: %s",
                       node_name)

    logger.info(
        "[INIT] Starting network interface discovery daemon for node: %s",
        node_name)
    logger.info("[INIT] Discovery interval: %d seconds", DISCOVERY_INTERVAL)
    logger.info("[INIT] Target namespace: %s", CV_INTERFACE_NAMESPACE)
    logger.info(
        "[INIT] Discovering both SR-IOV and non-SR-IOV physical interfaces")

    # Initialize Kubernetes client
    try:
        config.load_incluster_config()
        logger.info("[INIT] Loaded in-cluster Kubernetes config")
    except Exception:
        try:
            config.load_kube_config()
            logger.info("[INIT] Loaded kubeconfig")
        except Exception as e:
            logger.error("[INIT] Failed to load Kubernetes config: %s", e)
            sys.exit(1)

    api = client.CustomObjectsApi()
    v1_api = client.CoreV1Api()

    # Initialize caches
    cache = NodeStateCache()  # For NodeInterfaceState CR
    vf_cache = VFCache()  # For VF details across namespace moves

    # Main discovery loop
    while True:
        try:
            logger.debug("Running interface discovery on node %s", node_name)
            interfaces = discover_interfaces(vf_cache)

            # Create or update the NodeInterfaceState CR
            # Uses cache to avoid unnecessary API calls
            # Logging is handled inside create_or_update_nodestate based on changes
            create_or_update_nodestate(api, v1_api, node_name, interfaces,
                                       cache)

        except Exception as e:
            logger.error("[DISCOVERY] Error in discovery loop: %s",
                         e,
                         exc_info=True)

        # Wait before next discovery
        time.sleep(DISCOVERY_INTERVAL)


if __name__ == "__main__":
    main()
