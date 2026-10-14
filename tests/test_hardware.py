import json
import subprocess
import unittest
from unittest.mock import patch

import hardware


class HardwareTests(unittest.TestCase):
    def setUp(self):
        host = patch("hardware.wsl_gpu_inventory", return_value={"gpus": [], "warnings": []})
        host.start()
        self.addCleanup(host.stop)

    def test_nvidia_memory_and_stable_identity(self):
        result = subprocess.CompletedProcess([], 0, "GPU-id, Test GPU, 8192, 4096, 550.0\n", "")
        with (
            patch("hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("hardware.subprocess.run", return_value=result),
            patch("hardware.Path.glob", return_value=[]),
        ):
            inventory = hardware.gpu_inventory()
        self.assertEqual(inventory["gpus"][0]["vram_total_bytes"], 8192 * 1024 * 1024)
        self.assertEqual(inventory["gpus"][0]["vram_free_bytes"], 4096 * 1024 * 1024)
        self.assertNotIn("vram_free_bytes", hardware.stable_gpus(inventory)[0])
        self.assertEqual(hardware.stable_gpus(inventory)[0]["driver"], "550.0")

    def test_probe_failure_is_explicit(self):
        with (
            patch("hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("hardware.subprocess.run", side_effect=subprocess.TimeoutExpired("smi", 5)),
            patch("hardware.Path.glob", return_value=[]),
        ):
            result = hardware.gpu_inventory()
        self.assertEqual(result["gpus"], [])
        self.assertTrue(any("unavailable" in warning for warning in result["warnings"]))

    def test_invalid_csv_has_warning_not_false_memory(self):
        result = subprocess.CompletedProcess([], 0, "bad,data\n", "")
        with (
            patch("hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("hardware.subprocess.run", return_value=result),
            patch("hardware.Path.glob", return_value=[]),
        ):
            inventory = hardware.gpu_inventory()
        self.assertEqual(inventory["gpus"], [])
        self.assertTrue(inventory["warnings"])


class WslHardwareTests(unittest.TestCase):
    def test_windows_host_amd_is_detected_and_nvidia_is_not_duplicated(self):
        adapters = [
            {
                "PNPDeviceID": "PCI\\VEN_1002&DEV_13C0",
                "Name": "AMD Radeon(TM) Graphics",
                "AdapterCompatibility": "Advanced Micro Devices, Inc.",
                "AdapterRAM": 2 * 1024**3,
                "DriverVersion": "32.0",
            },
            {
                "PNPDeviceID": "PCI\\VEN_10DE&DEV_2B85",
                "Name": "Test NVIDIA",
                "AdapterCompatibility": "NVIDIA",
                "AdapterRAM": 4293918720,
                "DriverVersion": "32.0",
            },
        ]
        smi = subprocess.CompletedProcess([], 0, "GPU-id, Test NVIDIA, 32768, 4096, 550\n", "")
        cim = subprocess.CompletedProcess([], 0, json.dumps(adapters), "")
        with (
            patch("hardware.platform.release", return_value="microsoft-standard-WSL2"),
            patch("hardware.shutil.which", side_effect=lambda name: "/bin/" + name),
            patch("hardware.subprocess.run", side_effect=[smi, cim]),
            patch("hardware.Path.glob", return_value=[]),
        ):
            inventory = hardware.gpu_inventory()
        self.assertEqual(len(inventory["gpus"]), 2)
        nvidia, amd = inventory["gpus"]
        self.assertEqual(nvidia["vram_total_bytes"], 32768 * 1024**2)
        self.assertEqual(amd["vendor"], "AMD")
        self.assertEqual(amd["adapter_ram_reported_bytes"], 2 * 1024**3)
        self.assertIsNone(amd["vram_total_bytes"])
        self.assertIsNone(amd["wsl_inference_available"])
        self.assertEqual(amd["visibility"], "windows-host")
        self.assertTrue(any("32-bit" in warning for warning in inventory["warnings"]))

    def test_host_probe_failures_are_reported(self):
        for result in (
            subprocess.CompletedProcess([], 1, "", "access denied"),
            subprocess.CompletedProcess([], 0, "not JSON", ""),
            subprocess.CompletedProcess([], 0, "{}", ""),
            subprocess.CompletedProcess([], 0, '[{"Name":"GPU"}]', ""),
        ):
            with (
                self.subTest(result=result),
                patch("hardware.platform.release", return_value="microsoft-WSL2"),
                patch("hardware.shutil.which", return_value="/bin/powershell.exe"),
                patch("hardware.subprocess.run", return_value=result),
            ):
                inventory = hardware.wsl_gpu_inventory()
            self.assertEqual(inventory["gpus"], [])
            self.assertTrue(inventory["warnings"])

    def test_probe_timeout_and_missing_powershell(self):
        with (
            patch("hardware.platform.release", return_value="microsoft-WSL2"),
            patch("hardware.shutil.which", return_value="/bin/powershell.exe"),
            patch("hardware.subprocess.run", side_effect=subprocess.TimeoutExpired("ps", 10)),
        ):
            self.assertTrue(hardware.wsl_gpu_inventory()["warnings"])
        with (
            patch("hardware.platform.release", return_value="microsoft-WSL2"),
            patch("hardware.shutil.which", return_value=None),
            patch("hardware.Path.is_file", return_value=False),
        ):
            self.assertTrue(hardware.wsl_gpu_inventory()["warnings"])

    def test_linux_does_not_query_windows(self):
        with (
            patch("hardware.platform.release", return_value="6.8-linux"),
            patch("hardware.subprocess.run") as run,
        ):
            self.assertEqual(hardware.wsl_gpu_inventory(), {"gpus": [], "warnings": []})
            run.assert_not_called()

    def test_identical_gpu_names_match_one_for_one(self):
        native = [{"name": "GPU", "vendor": "NVIDIA", "id": "native"}]
        host = [
            {"name": "GPU", "vendor": "NVIDIA", "id": "host1"},
            {"name": "GPU", "vendor": "NVIDIA", "id": "host2"},
        ]
        hardware.merge_host_gpus(native, host)
        self.assertEqual([gpu["id"] for gpu in native], ["native", "host2"])


if __name__ == "__main__":
    unittest.main()
