"""
BioPrint Vision : Scanner Device Manager

Orchestrates physical hardware discovery, driver lifecycle, failover, and
biometric capture routing across optical, capacitive, serial, and virtual scanners.
"""

import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from services.scanner.base import (
    FingerprintScannerDriver,
    ScannerDeviceInfo,
    ScannerCaptureResult,
)
from services.scanner.drivers import (
    SimulatorScannerDriver,
    SecuGenScannerDriver,
    MantraScannerDriver,
    DigitalPersonaDriver,
    SerialGrowDriver,
)
from config import Config


class ScannerDeviceManager:
    """
    Central manager for biometric scanner hardware discovery and capture routing.
    """

    def __init__(self):
        self._drivers: Dict[str, FingerprintScannerDriver] = {}
        self._register_default_drivers()

    def _register_default_drivers(self):
        """Register supported hardware drivers and the high-fidelity simulator."""
        self.register_driver(SimulatorScannerDriver())
        self.register_driver(SecuGenScannerDriver())
        self.register_driver(MantraScannerDriver())
        self.register_driver(DigitalPersonaDriver())
        self.register_driver(SerialGrowDriver())

    def register_driver(self, driver: FingerprintScannerDriver):
        """Register or override a scanner driver."""
        self._drivers[driver.driver_id] = driver

    def get_driver(self, device_id: str) -> Optional[FingerprintScannerDriver]:
        """Retrieve a specific driver by device ID."""
        return self._drivers.get(device_id)

    def list_devices(self, probe_hardware: bool = True) -> List[ScannerDeviceInfo]:
        """
        Enumerate all registered scanner drivers and return their real-time status.
        Physical connected devices are listed first, followed by available simulators.
        """
        results: List[ScannerDeviceInfo] = []
        for driver in self._drivers.values():
            try:
                info = driver.get_device_info()
                results.append(info)
            except Exception as e:
                results.append(ScannerDeviceInfo(
                    device_id=driver.driver_id,
                    name=driver.driver_name,
                    vendor=driver.vendor,
                    device_type=driver.device_type,
                    is_connected=False,
                    status_text=f"Probe error: {str(e)}",
                    dpi=driver.dpi,
                    interface="Unknown",
                ))

        # Sort: connected physical devices first, then simulator, then offline devices
        def sort_key(d: ScannerDeviceInfo):
            if d.device_id == "simulator_sensor":
                return 1
            if d.is_connected:
                return 0
            return 2

        return sorted(results, key=sort_key)

    def get_active_device_id(self) -> str:
        """
        Determine the primary active device.
        Returns first connected physical scanner; falls back to 'simulator_sensor'.
        """
        devices = self.list_devices(probe_hardware=True)
        for d in devices:
            if d.is_connected and d.device_id != "simulator_sensor":
                return d.device_id
        return "simulator_sensor"

    def capture(
        self,
        device_id: Optional[str] = None,
        timeout_sec: float = 10.0,
        **kwargs
    ) -> ScannerCaptureResult:
        """
        Perform a live biometric capture from the specified or active device.
        """
        target_id = device_id or self.get_active_device_id()
        driver = self.get_driver(target_id)

        if not driver:
            return ScannerCaptureResult(
                success=False,
                device_id=target_id,
                device_name=f"Unknown Device ({target_id})",
                error_message=f"No driver registered for device '{target_id}'.",
            )

        # Execute capture
        res = driver.capture(timeout_sec=timeout_sec, **kwargs)

        # If a physical driver fails because hardware is absent, provide helpful guidance
        if not res.success and target_id != "simulator_sensor":
            # Check if simulator fallback was explicitly requested
            if kwargs.get("allow_simulator_fallback", False):
                sim_driver = self.get_driver("simulator_sensor")
                if sim_driver:
                    return sim_driver.capture(**kwargs)

        return res

    def capture_and_save(
        self,
        device_id: Optional[str] = None,
        output_folder: Optional[Path] = None,
        **kwargs
    ) -> Tuple[ScannerCaptureResult, Optional[Path]]:
        """
        Trigger hardware capture and persist the resulting image to disk.

        Returns:
            (ScannerCaptureResult, saved_file_path)
        """
        res = self.capture(device_id=device_id, **kwargs)
        if not res.success or res.pil_image is None:
            return res, None

        folder = output_folder or Config.FINGERPRINT_FOLDER
        folder.mkdir(parents=True, exist_ok=True)

        uid = str(uuid.uuid4())[:8]
        filename = f"scanner_capture_{uid}_{res.device_id}.bmp"
        file_path = folder / filename

        # Save standard uncompressed BMP for high biometric fidelity
        res.pil_image.save(str(file_path), format="BMP")
        return res, file_path


# Global singleton instance
_scanner_manager: Optional[ScannerDeviceManager] = None


def get_scanner_manager() -> ScannerDeviceManager:
    """Retrieve or initialize the global ScannerDeviceManager singleton."""
    global _scanner_manager
    if _scanner_manager is None:
        _scanner_manager = ScannerDeviceManager()
    return _scanner_manager
