"""
BioPrint Vision : Physical Fingerprint Scanner SDK: Base Architecture & Contracts

Defines the abstract interface, device metadata structures, and capture result
models for physical biometric fingerprint sensors and hardware bridges.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import io
import base64
from PIL import Image


@dataclass
class ScannerDeviceInfo:
    """Represents metadata and real-time connectivity status of a fingerprint scanner."""
    device_id: str
    name: str
    vendor: str
    device_type: str            # e.g., 'optical_usb', 'capacitive_usb', 'virtual_500dpi', 'serial_uart'
    is_connected: bool
    status_text: str            # e.g., 'Ready', 'Sensor Offline', 'Service Not Running'
    serial_number: Optional[str] = None
    firmware_version: Optional[str] = None
    dpi: int = 500
    description: str = ""
    interface: str = "USB"      # e.g., 'USB', 'Serial / UART', 'Localhost Bridge'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScannerCaptureResult:
    """Holds the raw biometric impression, converted PIL Image, and quality metrics."""
    success: bool
    device_id: str
    device_name: str
    pil_image: Optional[Image.Image] = None
    image_bytes: Optional[bytes] = None
    raw_format: str = "BMP"     # Typically BMP for biometric sensors
    width: int = 0
    height: int = 0
    dpi: int = 500
    quality_score: float = 0.0  # 0.00 to 1.00
    quality_status: str = "Unknown"  # 'Good', 'Fair', 'Poor'
    capture_time_ms: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self, include_base64: bool = True) -> Dict[str, Any]:
        res = {
            "success": self.success,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "quality_score": round(self.quality_score, 2),
            "quality_status": self.quality_status,
            "capture_time_ms": round(self.capture_time_ms, 1),
            "error_message": self.error_message,
        }
        if include_base64 and self.image_bytes:
            b64_str = base64.b64encode(self.image_bytes).decode("ascii")
            res["image_base64"] = f"data:image/bmp;base64,{b64_str}"
        return res


class FingerprintScannerDriver(ABC):
    """
    Abstract Base Class for all physical and simulated fingerprint scanner drivers.
    Each supported hardware vendor implements this interface.
    """

    driver_id: str = "base_driver"
    driver_name: str = "Base Scanner Driver"
    vendor: str = "Generic"
    device_type: str = "optical_usb"
    dpi: int = 500
    supported_models: List[str] = []

    @abstractmethod
    def check_availability(self) -> bool:
        """Probe whether the scanner device or its local driver/service is reachable."""
        pass

    @abstractmethod
    def get_device_info(self) -> ScannerDeviceInfo:
        """Return standardized device metadata and connectivity status."""
        pass

    @abstractmethod
    def capture(self, timeout_sec: float = 10.0, **kwargs) -> ScannerCaptureResult:
        """
        Trigger an image capture cycle on the physical sensor.

        Args:
            timeout_sec: Maximum seconds to wait for finger placement and scan.
            **kwargs: Vendor-specific parameters.

        Returns:
            ScannerCaptureResult containing the captured PIL Image and metadata.
        """
        pass
