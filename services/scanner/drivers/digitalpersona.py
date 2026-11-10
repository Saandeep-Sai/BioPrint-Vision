"""
BioPrint Vision : DigitalPersona U.are.U Fingerprint Scanner Driver

Integrates with DigitalPersona U.are.U series (4500, 5160, 5300) optical USB
fingerprint readers via the DigitalPersona Web Client / One Touch service.
"""

import io
import time
import json
import base64
import urllib.request
from typing import Optional
from PIL import Image

from services.scanner.base import (
    FingerprintScannerDriver,
    ScannerDeviceInfo,
    ScannerCaptureResult,
)
from preprocessing.quality import check_image_quality


class DigitalPersonaDriver(FingerprintScannerDriver):
    """
    Driver for DigitalPersona U.are.U 4500 / 5160 / 5300 optical USB biometric scanners.
    Default local service port: 52181.
    """

    driver_id = "digitalpersona_uareu"
    driver_name = "DigitalPersona U.are.U Series (Optical USB)"
    vendor = "HID Global / DigitalPersona"
    device_type = "optical_usb"
    dpi = 500
    supported_models = [
        "U.are.U 4500 Reader",
        "U.are.U 5160 Reader",
        "U.are.U 5300 Reader",
    ]

    DEFAULT_URL = "http://127.0.0.1:52181"

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or self.DEFAULT_URL
        self._is_ready = False

    def check_availability(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.endpoint_url}/api/v1/device", method="GET")
            with urllib.request.urlopen(req, timeout=0.4) as resp:
                if resp.status == 200:
                    self._is_ready = True
                    return True
        except Exception:
            self._is_ready = False
        return False

    def get_device_info(self) -> ScannerDeviceInfo:
        is_avail = self.check_availability()
        return ScannerDeviceInfo(
            device_id=self.driver_id,
            name=self.driver_name,
            vendor=self.vendor,
            device_type=self.device_type,
            is_connected=is_avail,
            status_text="Ready (Reader Connected)" if is_avail else "Service / Reader Offline",
            serial_number="DP-4500-USB",
            firmware_version="v3.2",
            dpi=self.dpi,
            description="512 DPI optical biometric scanner with blue illumination and silicone coating",
            interface="DigitalPersona Web Client (localhost:52181)",
        )

    def capture(self, timeout_sec: float = 10.0, **kwargs) -> ScannerCaptureResult:
        start_time = time.perf_counter()
        if not self.check_availability():
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=(
                    "DigitalPersona service not reachable on localhost:52181. "
                    "Ensure DigitalPersona Web SDK or DigitalPersona Workstation service is active."
                ),
            )

        try:
            payload = json.dumps({"Timeout": int(timeout_sec * 1000)}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint_url}/api/v1/capture",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout_sec + 2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            b64_img = data.get("Image") or data.get("BMPBase64")
            if not b64_img:
                return ScannerCaptureResult(
                    success=False,
                    device_id=self.driver_id,
                    device_name=self.driver_name,
                    error_message=data.get("Error", "No image received from DigitalPersona reader."),
                )

            bmp_bytes = base64.b64decode(b64_img)
            pil_img = Image.open(io.BytesIO(bmp_bytes)).convert("RGB")
            q_result = check_image_quality(pil_img)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ScannerCaptureResult(
                success=True,
                device_id=self.driver_id,
                device_name=self.driver_name,
                pil_image=pil_img,
                image_bytes=bmp_bytes,
                raw_format="BMP",
                width=pil_img.width,
                height=pil_img.height,
                dpi=self.dpi,
                quality_score=q_result.get("score", 0.90),
                quality_status=q_result.get("status", "Good"),
                capture_time_ms=elapsed_ms,
            )

        except Exception as e:
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=f"DigitalPersona capture error: {str(e)}",
            )
