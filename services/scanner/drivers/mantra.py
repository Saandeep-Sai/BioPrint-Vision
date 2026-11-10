"""
BioPrint Vision : Mantra MFS100 Fingerprint Scanner Driver

Integrates with Mantra MFS100 optical USB biometric scanners via the
Mantra RD Service / Web Client API (localhost:11100 / 11101).
"""

import io
import time
import base64
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any
from PIL import Image

from services.scanner.base import (
    FingerprintScannerDriver,
    ScannerDeviceInfo,
    ScannerCaptureResult,
)
from preprocessing.quality import check_image_quality


class MantraScannerDriver(FingerprintScannerDriver):
    """
    Driver for Mantra MFS100 USB optical fingerprint scanners via localhost RD Service.
    Standard ports: 11100, 11101, 11102.
    """

    driver_id = "mantra_mfs100"
    driver_name = "Mantra MFS100 Optical Scanner"
    vendor = "Mantra Softech India"
    device_type = "optical_usb"
    dpi = 500
    supported_models = ["MFS100 V54", "MFS100 V61", "MFS500"]

    DEFAULT_PORTS = [11100, 11101, 11102]

    def __init__(self, port: Optional[int] = None):
        self.port = port
        self._active_url: Optional[str] = None
        self._device_info: Dict[str, Any] = {}

    def _find_service(self) -> Optional[str]:
        """Probe localhost ports to find active Mantra RD Service."""
        ports = [self.port] if self.port else self.DEFAULT_PORTS
        for p in ports:
            if not p:
                continue
            url = f"http://127.0.0.1:{p}"
            try:
                req = urllib.request.Request(f"{url}/rd/info", method="GET")
                with urllib.request.urlopen(req, timeout=0.4) as resp:
                    if resp.status == 200:
                        content = resp.read().decode("utf-8", errors="ignore")
                        self._active_url = url
                        self._parse_device_info(content)
                        return url
            except Exception:
                continue
        self._active_url = None
        return None

    def _parse_device_info(self, xml_or_json: str):
        try:
            if xml_or_json.strip().startswith("<"):
                root = ET.fromstring(xml_or_json)
                self._device_info = {
                    "status": root.attrib.get("status", "READY"),
                    "info": root.attrib.get("info", "MFS100"),
                }
        except Exception:
            pass

    def check_availability(self) -> bool:
        return self._find_service() is not None

    def get_device_info(self) -> ScannerDeviceInfo:
        is_avail = self.check_availability()
        return ScannerDeviceInfo(
            device_id=self.driver_id,
            name=self.driver_name,
            vendor=self.vendor,
            device_type=self.device_type,
            is_connected=is_avail,
            status_text="Ready (Sensor Active)" if is_avail else "RD Service Offline",
            serial_number="MFS100-USB",
            firmware_version="v54",
            dpi=self.dpi,
            description="STQC certified 500 DPI optical fingerprint sensor with scratch-free optical platen",
            interface="Mantra RD Service (localhost:11100)",
        )

    def capture(self, timeout_sec: float = 10.0, **kwargs) -> ScannerCaptureResult:
        start_time = time.perf_counter()
        active_url = self._active_url or self._find_service()

        if not active_url:
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=(
                    "Mantra MFS100 RD Service not running. Install Mantra MFS100 driver & RD service "
                    "and connect the USB scanner."
                ),
            )

        try:
            # Standard Mantra RD Service CAPTURE command
            capture_xml = (
                f'<PidOptions ver="1.0">'
                f'<Opts fCount="1" fType="0" iCount="0" pCount="0" format="0" pidVer="2.0" timeout="{int(timeout_sec * 1000)}" env="P" />'
                f'</PidOptions>'
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{active_url}/rd/capture",
                data=capture_xml,
                headers={"Content-Type": "text/xml"},
                method="CAPTURE",
            )

            with urllib.request.urlopen(req, timeout=timeout_sec + 2.0) as resp:
                resp_text = resp.read().decode("utf-8", errors="ignore")

            # Parse XML response
            root = ET.fromstring(resp_text)
            resp_elem = root.find(".//Resp")
            err_code = resp_elem.attrib.get("errCode", "0") if resp_elem is not None else "0"
            if err_code != "0":
                err_info = resp_elem.attrib.get("errInfo", f"Mantra error code {err_code}")
                return ScannerCaptureResult(
                    success=False,
                    device_id=self.driver_id,
                    device_name=self.driver_name,
                    error_message=err_info,
                )

            # Extract base64 biometric data
            data_elem = root.find(".//Data")
            if data_elem is None or not data_elem.text:
                return ScannerCaptureResult(
                    success=False,
                    device_id=self.driver_id,
                    device_name=self.driver_name,
                    error_message="Mantra response did not contain biometric data.",
                )

            bmp_bytes = base64.b64decode(data_elem.text.strip())
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
                quality_score=q_result.get("score", 0.88),
                quality_status=q_result.get("status", "Good"),
                capture_time_ms=elapsed_ms,
            )

        except Exception as e:
            return ScannerCaptureResult(
                success=False,
                device_id=self.driver_id,
                device_name=self.driver_name,
                error_message=f"Mantra MFS100 capture error: {str(e)}",
            )
