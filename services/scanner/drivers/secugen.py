"""
BioPrint Vision : SecuGen Fingerprint Scanner Driver

Integrates with SecuGen optical fingerprint scanners (Hamster Pro 20, Hamster Plus,
Hamster IV, Unity series) via the SecuGen WebAPI / FDx local service bridge.
"""

import io
import time
import json
import base64
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from PIL import Image

from services.scanner.base import (
    FingerprintScannerDriver,
    ScannerDeviceInfo,
    ScannerCaptureResult,
)
from preprocessing.quality import check_image_quality


class SecuGenScannerDriver(FingerprintScannerDriver):
    """
    Driver for SecuGen Hamster USB optical scanners via localhost WebAPI service.
    Default ports: 8084 (HTTP), 8443 (HTTPS).
    """

    driver_id = "secugen_webapi"
    driver_name = "SecuGen Hamster Series (USB Optical)"
    vendor = "SecuGen Corporation"
    device_type = "optical_usb"
    dpi = 500
    supported_models = [
        "Hamster Pro 20 (HUPx)",
        "Hamster Plus (HSDU03P)",
        "Hamster IV (HFDU04)",
        "Hamster Air",
        "Unity 20 Bluetooth",
    ]

    DEFAULT_HOSTS = ["http://127.0.0.1:8084", "https://127.0.0.1:8443"]

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url
        self._active_url: Optional[str] = None
        self._last_device_info: Dict[str, Any] = {}

    def _probe_service(self) -> Optional[str]:
        """Check if SecuGen local service bridge is actively listening."""
        candidates = [self.endpoint_url] if self.endpoint_url else self.DEFAULT_HOSTS
        for url in candidates:
            if not url:
                continue
            try:
                # Test connection with short 400ms timeout
                probe_req = urllib.request.Request(
                    f"{url}/sgisvc/deviceinfo",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(probe_req, timeout=0.5) as resp:
                    if resp.status in (200, 204):
                        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                        self._last_device_info = data
                        self._active_url = url
                        return url
            except Exception:
                # Fallback probe to root
                try:
                    with urllib.request.urlopen(f"{url}/", timeout=0.4) as resp:
                        if resp.status == 200:
                            self._active_url = url
                            return url
                except Exception:
                    continue
        self._active_url = None
        return None

    def check_availability(self) -> bool:
        """Returns True if SecuGen local WebAPI service and scanner are detected."""
        return self._probe_service() is not None

    def get_device_info(self) -> ScannerDeviceInfo:
        is_avail = self.check_availability()
        serial = self._last_device_info.get("DeviceSN") or self._last_device_info.get("SerialNumber")
        model = self._last_device_info.get("DeviceModel", "Hamster Pro 20")

        return ScannerDeviceInfo(
            device_id=self.driver_id,
            name=f"SecuGen {model}",
            vendor=self.vendor,
            device_type=self.device_type,
            is_connected=is_avail,
            status_text="Ready (Hardware Connected)" if is_avail else "Service / Hardware Not Detected",
            serial_number=serial if is_avail else "Not Connected",
            firmware_version=self._last_device_info.get("FWVersion", "Unknown"),
            dpi=self.dpi,
            description="High-precision 500 DPI optical sensor with Auto-On touch detection and Fake Finger Rejection",
            interface="SecuGen FDx WebAPI (localhost:8084)",
        )

    def capture(self, timeout_sec: float = 10.0, **kwargs) -> ScannerCaptureResult:
        start_time = time.perf_counter()
        active_url = self._active_url or self._probe_service()

        if not active_url:
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=(
                    "SecuGen device service not detected. Ensure SecuGen WebAPI / FDx service "
                    "is installed and running on localhost:8084 or 8443, and the USB scanner is plugged in."
                ),
            )

        try:
            # SecuGen WebAPI Capture Payload
            payload = json.dumps({
                "Timeout": int(timeout_sec * 1000),
                "Quality": kwargs.get("min_quality", 50),
                "templateFormat": "ISO",
                "imageWSQRate": "0",
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{active_url}/sgisvc/capture",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=timeout_sec + 2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            err_code = data.get("ErrorCode", 0)
            if err_code != 0:
                err_msg = data.get("ErrorDescription") or f"SecuGen driver returned error code {err_code}"
                return ScannerCaptureResult(
                    success=False,
                    device_id=self.driver_id,
                    device_name=self.driver_name,
                    error_message=err_msg,
                )

            # Extract base64 BMP image
            bmp_b64 = data.get("BMPBase64") or data.get("ImageBase64")
            if not bmp_b64:
                return ScannerCaptureResult(
                    success=False,
                    device_id=self.driver_id,
                    device_name=self.driver_name,
                    error_message="SecuGen response did not contain image data.",
                )

            bmp_bytes = base64.b64decode(bmp_b64)
            pil_img = Image.open(io.BytesIO(bmp_bytes)).convert("RGB")

            # Assess quality
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

        except urllib.error.URLError as e:
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=f"Communication error with SecuGen WebAPI service: {e.reason}",
            )
        except Exception as e:
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=f"SecuGen capture failed: {str(e)}",
            )
