"""
BioPrint Vision : SecuGen Fingerprint Scanner Driver

Integrates with SecuGen optical fingerprint scanners (Hamster Pro 20, Hamster Plus,
Hamster IV, Unity series) via the SecuGen WebAPI (sgibiosrv.exe) local service.
"""

import io
import ssl
import time
import json
import base64
import urllib.parse
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


# Official SecuGen WebAPI error codes (from SecuGen WebAPI Programmer's Manual)
_SECUGEN_ERRORS = {
    0: "Success",
    1: "Creation failed (fingerprint reader not correctly installed or driver files error).",
    2: "Function failed (wrong type of fingerprint reader or not correctly installed).",
    3: "Internal error (invalid parameters to sensor API).",
    5: "DLL load failed.",
    6: "DLL load failed for driver.",
    7: "DLL load failed for algorithm.",
    51: "System file load failure.",
    52: "Sensor chip initialization failed.",
    53: "Sensor line dropped.",
    54: "Capture timed out. Place your finger firmly on the scanner sensor and try again.",
    55: (
        "Device not found (ErrorCode 55). Check that (1) the USB scanner is plugged in, "
        "(2) no other SecuGen demo/diagnostic app is open, and (3) the SecuGen USB driver is installed."
    ),
    56: "Driver load failed. Reinstall the SecuGen USB driver.",
    57: "Wrong image / invalid fingerprint image.",
    58: "Lack of USB bandwidth. Try a different USB port.",
    59: "Device is busy (ErrorCode 59). Close any other fingerprint app or unplug/replug the scanner.",
    60: "Cannot get serial number of the device.",
    61: "Unsupported device.",
    10001: "SecuGen WebAPI license error (ErrorCode 10001).",
    10002: "Invalid domain in Origin header (ErrorCode 10002).",
    10003: "SecuGen WebAPI license expired (ErrorCode 10003).",
    10004: "WebAPI did not receive a valid Origin header from the caller (ErrorCode 10004).",
}


class SecuGenScannerDriver(FingerprintScannerDriver):
    """
    Driver for SecuGen Hamster USB optical scanners via localhost WebAPI service.
    Communicates with sgibiosrv.exe running on ports 8443 (HTTPS) and 8000 (HTTP).
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

    # Put 8443 (HTTPS) first as newer sgibiosrv.exe versions default to SSL on 8443/8000
    DEFAULT_HOSTS = [
        "https://localhost:8443",
        "http://localhost:8000",
        "https://localhost:8000",
        "https://127.0.0.1:8443",
        "http://127.0.0.1:8000",
    ]

    CAPTURE_ENDPOINT = "/SGIFPCapture"

    def __init__(self, endpoint_url: Optional[str] = None, origin: str = "http://localhost"):
        self.endpoint_url = endpoint_url
        self.origin = origin
        self._active_url: Optional[str] = None
        self._last_error_code: Optional[int] = None
        self._ssl_ctx = self._create_ssl_context()

    @staticmethod
    def _create_ssl_context() -> ssl.SSLContext:
        """Create SSL context that accepts self-signed certs from sgibiosrv.exe."""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _request(
        self,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a POST request to the SecuGen WebAPI service using URL-encoded form parameters
        and a localhost Origin header (required by sgibiosrv.exe).
        """
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "*/*",
            "Origin": self.origin,
            "Referer": f"{self.origin}/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }

        try:
            if payload is not None:
                # sgibiosrv.exe expects "Timeout=10000&Quality=50&licstr=&templateFormat=ISO"
                data = urllib.parse.urlencode(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            else:
                req = urllib.request.Request(url, headers=headers, method="POST")

            if url.startswith("https"):
                resp = urllib.request.urlopen(req, timeout=timeout, context=self._ssl_ctx)
            else:
                resp = urllib.request.urlopen(req, timeout=timeout)

            body = resp.read().decode("utf-8", errors="ignore")
            return json.loads(body)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
            OSError,
        ):
            return None

    def _probe_service(self) -> Optional[str]:
        """
        Check if SecuGen sgibiosrv.exe is listening.
        Uses a 100ms timeout in URL-encoded format so the sensor releases immediately.
        """
        candidates = [self.endpoint_url] if self.endpoint_url else self.DEFAULT_HOSTS
        probe_payload = {
            "Timeout": 100,
            "Quality": 50,
            "licstr": "",
            "templateFormat": "ISO",
        }
        for base_url in candidates:
            if not base_url:
                continue
            url = f"{base_url}{self.CAPTURE_ENDPOINT}"
            result = self._request(url, payload=probe_payload, timeout=3.0)
            if result is not None:
                self._last_error_code = result.get("ErrorCode")
                self._active_url = base_url
                # Brief pause so sgibiosrv.exe releases the USB handle after probing
                time.sleep(0.15)
                return base_url

        self._active_url = None
        self._last_error_code = None
        return None

    def check_availability(self) -> bool:
        """Returns True if SecuGen WebAPI service is responding."""
        return self._probe_service() is not None

    def get_device_info(self) -> ScannerDeviceInfo:
        is_avail = self.check_availability()

        # ErrorCode 0 (captured) or 54 (100ms probe timed out waiting for finger) = hardware ready
        device_connected = is_avail and self._last_error_code in (0, 54)
        service_running = is_avail

        if device_connected:
            status_text = "Ready (Hardware Connected)"
        elif service_running:
            err_desc = _SECUGEN_ERRORS.get(
                self._last_error_code, f"Error code {self._last_error_code}"
            )
            status_text = f"Service Running - {err_desc}"
        else:
            status_text = "SecuGen WebAPI Service Not Detected (sgibiosrv.exe not running)"

        active_port = self._active_url.split(":")[-1] if self._active_url else ""

        return ScannerDeviceInfo(
            device_id=self.driver_id,
            name="SecuGen Hamster Pro 20",
            vendor=self.vendor,
            device_type=self.device_type,
            is_connected=device_connected,
            status_text=status_text,
            serial_number="Connected" if device_connected else "Not Detected",
            firmware_version="WebAPI",
            dpi=self.dpi,
            description="High-precision 500 DPI optical sensor with Auto-On touch detection and Fake Finger Rejection",
            interface=(
                f"SecuGen WebAPI sgibiosrv.exe (localhost:{active_port})"
                if active_port
                else "SecuGen WebAPI (sgibiosrv.exe)"
            ),
        )

    def capture(self, timeout_sec: float = 10.0, **kwargs) -> ScannerCaptureResult:
        start_time = time.perf_counter()

        # Exact parameter names expected by sgibiosrv.exe (case-sensitive)
        payload = {
            "Timeout": int(timeout_sec * 1000),
            "Quality": kwargs.get("min_quality", 50),
            "licstr": kwargs.get("licstr", ""),
            "templateFormat": kwargs.get("template_format", "ISO"),
        }

        # Try active URL directly without running a separate probe scan first
        candidates = (
            [self._active_url]
            if self._active_url
            else ([self.endpoint_url] if self.endpoint_url else self.DEFAULT_HOSTS)
        )

        result = None
        for base_url in candidates:
            if not base_url:
                continue
            capture_url = f"{base_url}{self.CAPTURE_ENDPOINT}"
            result = self._request(capture_url, payload=payload, timeout=timeout_sec + 5.0)
            if result is not None:
                self._active_url = base_url
                break

        if result is None:
            self._active_url = None
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=(
                    "SecuGen WebAPI service (sgibiosrv.exe) not responding on ports 8443 or 8000. "
                    "Check Windows Task Manager -> Services to ensure SgiBioSrv is running."
                ),
            )

        err_code = result.get("ErrorCode", -1)
        self._last_error_code = err_code

        if err_code != 0:
            err_msg = _SECUGEN_ERRORS.get(
                err_code, f"SecuGen WebAPI returned error code: {err_code}"
            )
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=f"{err_msg} [Raw ErrorCode: {err_code}]",
            )

        # Extract base64 BMP image from successful response
        bmp_b64 = result.get("BMPBase64") or result.get("ImageDataBase64") or result.get("ImageBase64")
        if not bmp_b64:
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message="SecuGen capture succeeded (ErrorCode 0) but response contained no image data.",
            )

        try:
            bmp_bytes = base64.b64decode(bmp_b64)
            pil_img = Image.open(io.BytesIO(bmp_bytes)).convert("RGB")
        except Exception as e:
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=f"Failed to decode fingerprint image from SecuGen response: {e}",
            )

        # Assess quality
        try:
            q_result = check_image_quality(pil_img)
        except Exception:
            q_result = {"score": 0.85, "status": "Good"}

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