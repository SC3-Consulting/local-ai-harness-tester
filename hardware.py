"""Best-effort local hardware inventory without importing GPU frameworks."""

import csv
import io
import json
import platform
import shutil
import subprocess
from pathlib import Path


def wsl_gpu_inventory():
    """Discover host adapters, without assuming they are usable for WSL inference."""
    if "microsoft" not in platform.release().lower():
        return {"gpus": [], "warnings": []}
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        fallback = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
        if fallback.is_file():
            powershell = str(fallback)
    if powershell is None:
        return {
            "gpus": [],
            "warnings": ["WSL host GPU discovery unavailable: Windows PowerShell not accessible."],
        }
    command = (
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "@(Get-CimInstance Win32_VideoController -ErrorAction Stop | "
        "Select-Object PNPDeviceID,Name,AdapterCompatibility,AdapterRAM,DriverVersion) | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(f"PowerShell exited {result.returncode}: {result.stderr.strip()}")
        adapters = json.loads(result.stdout.lstrip("\ufeff"))
        if not isinstance(adapters, list):
            raise ValueError("Expected an array of Windows display adapters")
        devices = []
        for adapter in adapters:
            if not isinstance(adapter, dict):
                raise ValueError("Invalid Windows display adapter record")
            device_id, name = adapter.get("PNPDeviceID"), adapter.get("Name")
            if not isinstance(device_id, str) or not isinstance(name, str):
                raise ValueError("Windows display adapter is missing its ID or name")
            vendor = adapter.get("AdapterCompatibility")
            if "VEN_1002" in device_id.upper():
                vendor = "AMD"
            elif "VEN_10DE" in device_id.upper():
                vendor = "NVIDIA"
            elif "VEN_8086" in device_id.upper():
                vendor = "Intel"
            reported = adapter.get("AdapterRAM")
            if reported is not None and (type(reported) is not int or reported < 0):
                raise ValueError("Invalid Windows AdapterRAM value")
            devices.append(
                {
                    "id": device_id,
                    "name": name,
                    "vendor": vendor,
                    "vram_total_bytes": None,
                    "vram_free_bytes": None,
                    "adapter_ram_reported_bytes": reported,
                    "driver": adapter.get("DriverVersion"),
                    "source": "windows-cim",
                    "visibility": "windows-host",
                    "wsl_inference_available": None,
                }
            )
        return {"gpus": devices, "warnings": []}
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {"gpus": [], "warnings": [f"WSL host GPU discovery unavailable: {exc}"]}


def merge_host_gpus(devices, host_devices):
    """Match host names one-for-one while preserving native memory measurements."""
    unmatched = list(devices)
    for host in host_devices:
        match = next(
            (
                device
                for device in unmatched
                if device["name"].casefold() == host["name"].casefold()
                and device["vendor"] in (host["vendor"], "0x1002", "0x10de", "0x8086")
            ),
            None,
        )
        if match is not None:
            unmatched.remove(match)
        else:
            devices.append(host)


def gpu_inventory():
    """
    Inventory available GPUs on the local system.

    Returns:
        dict: GPU devices and any warnings encountered during inventory.
    """
    candidates = [shutil.which("nvidia-smi")]
    wsl_smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    if candidates[0] is None and wsl_smi.is_file():
        candidates[0] = str(wsl_smi)
    warnings = []
    devices = []
    if candidates[0]:
        try:
            result = subprocess.run(
                [
                    candidates[0],
                    "--query-gpu=uuid,name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                warnings.append(f"nvidia-smi failed: {result.stderr.strip()}")
            else:
                for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
                    if len(row) != 5:
                        raise ValueError("Unexpected nvidia-smi CSV response")
                    devices.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "vendor": "NVIDIA",
                            "vram_total_bytes": int(float(row[2]) * 1024 * 1024),
                            "vram_free_bytes": int(float(row[3]) * 1024 * 1024),
                            "driver": row[4],
                        }
                    )
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            warnings.append(f"NVIDIA inventory unavailable: {exc}")
            devices = []
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        vendor, product = device / "vendor", device / "device"
        if not vendor.is_file() or not product.is_file():
            continue
        vendor_id = vendor.read_text(encoding="utf-8").strip()
        if vendor_id == "0x10de" and any(g["vendor"] == "NVIDIA" for g in devices):
            continue
        total = device / "mem_info_vram_total"
        used = device / "mem_info_vram_used"
        try:
            total_bytes = int(total.read_text(encoding="utf-8")) if total.is_file() else None
            used_bytes = int(used.read_text(encoding="utf-8")) if used.is_file() else None
        except (OSError, ValueError) as exc:
            warnings.append(f"VRAM unavailable for {device}: {exc}")
            total_bytes, used_bytes = None, None
        devices.append(
            {
                "id": str(device.resolve()),
                "name": product.read_text(encoding="utf-8").strip(),
                "vendor": vendor_id,
                "vram_total_bytes": total_bytes,
                "vram_free_bytes": (
                    total_bytes - used_bytes
                    if total_bytes is not None and used_bytes is not None
                    else None
                ),
                "driver": (
                    (device / "driver").resolve().name if (device / "driver").exists() else None
                ),
            }
        )
    host = wsl_gpu_inventory()
    warnings.extend(host["warnings"])
    merge_host_gpus(devices, host["gpus"])
    if any(device.get("source") == "windows-cim" for device in devices):
        warnings.append(
            "Windows host adapters may not be usable by WSL inference runtimes. "
            "AdapterRAM is a 32-bit Windows-reported value, not verified dedicated VRAM "
            "or shared-memory capacity; total/free VRAM are unknown for host-only adapters."
        )
    if not devices:
        warnings.append("No GPU telemetry detected; this does not prove no GPU is present.")
    return {"gpus": devices, "warnings": warnings}


def memory_inventory():
    """
    Inventory the system's RAM.

    Returns:
        dict: A dictionary containing the total and available RAM in bytes.
    """
    values = {}
    source = Path("/proc/meminfo")
    if source.is_file():
        for line in source.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                values[key] = int(value.split()[0]) * 1024
    return {
        "ram_total_bytes": values.get("MemTotal"),
        "ram_available_bytes": values.get("MemAvailable"),
    }


def stable_gpus(inventory):
    """
    Extract stable GPU information from the inventory, excluding volatile fields like VRAM usage.

    Args:
        inventory (dict): The GPU inventory obtained from `gpu_inventory()`.

    Returns:
        list: A list of dictionaries containing stable GPU information.
    """
    return [
        {key: value for key, value in gpu.items() if key != "vram_free_bytes"}
        for gpu in inventory["gpus"]
    ]
