"""
BioPrint Vision : Hardware Simulator Scanner Driver

Provides a realistic virtual 500 DPI optical fingerprint sensor for development,
automated testing, continuous integration, and workstations where a physical USB
scanner is not currently plugged in.
"""

import io
import time
import random
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from services.scanner.base import (
    FingerprintScannerDriver,
    ScannerDeviceInfo,
    ScannerCaptureResult,
)
from preprocessing.quality import check_image_quality


class SimulatorScannerDriver(FingerprintScannerDriver):
    """
    Virtual plug-and-play biometric optical scanner.
    Simulates real-world sensor platen physics, optical noise, and touch latency.
    """

    driver_id = "simulator_sensor"
    driver_name = "BioPrint Virtual Optical Scanner (Simulated Hardware)"
    vendor = "BioPrint Biometrics"
    device_type = "virtual_500dpi"
    dpi = 500
    supported_models = ["BioPrint V-Scan 500", "Virtual Platen Pro"]

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"
        self._sample_paths: List[Path] = []
        self._refresh_sample_pool()

    def _refresh_sample_pool(self):
        """Locate genuine dataset impressions to simulate hardware platen captures."""
        if self.data_dir.exists():
            bmp_files = list(self.data_dir.rglob("*.BMP")) + list(self.data_dir.rglob("*.bmp"))
            self._sample_paths = bmp_files[:500] if bmp_files else []

    def check_availability(self) -> bool:
        """Simulator is always available to ensure uninterrupted plug-and-play development."""
        return True

    def get_device_info(self) -> ScannerDeviceInfo:
        return ScannerDeviceInfo(
            device_id=self.driver_id,
            name=self.driver_name,
            vendor=self.vendor,
            device_type=self.device_type,
            is_connected=True,
            status_text="Ready (Sensor Active)",
            serial_number="BP-SIM-90210-500DPI",
            firmware_version="v2.4.1-sim",
            dpi=self.dpi,
            description="500 DPI optical platen simulation with real-time ridge capture and noise injection",
            interface="Virtual USB Emulation",
        )

    def capture(self, timeout_sec: float = 10.0, **kwargs) -> ScannerCaptureResult:
        """
        Simulate a hardware optical capture cycle:
          1. Wait for simulated finger placement (simulated ADC latency: ~350-550ms)
          2. Load base impression or synthesize ridges
          3. Apply optical sensor characteristics (slight platen noise & contrast)
          4. Return standardized 500 DPI BMP
        """
        start_time = time.perf_counter()

        # Simulate hardware touch sensing & optical exposure latency
        simulate_latency = kwargs.get("simulate_latency", True)
        if simulate_latency:
            time.sleep(random.uniform(0.35, 0.60))

        # 1. Acquire base impression
        base_img: Optional[Image.Image] = None
        sample_path: Optional[str] = kwargs.get("sample_path")

        if sample_path and Path(sample_path).exists():
            try:
                base_img = Image.open(sample_path).convert("L")
            except Exception:
                base_img = None

        if base_img is None and self._sample_paths:
            chosen_path = random.choice(self._sample_paths)
            try:
                base_img = Image.open(chosen_path).convert("L")
            except Exception:
                base_img = None

        if base_img is None:
            # Fallback: synthesize realistic 500 DPI optical fingerprint ridges
            base_img = self._synthesize_fallback_ridges()

        # 2. Simulate optical sensor platen effects (500 DPI sensor characteristics)
        processed_img = self._apply_sensor_characteristics(base_img)

        # 3. Assess quality using system quality module
        q_result = check_image_quality(processed_img)

        # 4. Serialize to standard uncompressed BMP buffer
        buf = io.BytesIO()
        processed_img.save(buf, format="BMP")
        bmp_bytes = buf.getvalue()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ScannerCaptureResult(
            success=True,
            device_id=self.driver_id,
            device_name=self.driver_name,
            pil_image=processed_img.convert("RGB"),
            image_bytes=bmp_bytes,
            raw_format="BMP",
            width=processed_img.width,
            height=processed_img.height,
            dpi=self.dpi,
            quality_score=q_result.get("score", 0.85),
            quality_status=q_result.get("status", "Good"),
            capture_time_ms=elapsed_ms,
            error_message=None,
        )

    def _apply_sensor_characteristics(self, img: Image.Image) -> Image.Image:
        """Inject optical sensor noise, slight platen illumination gradient, and 500 DPI scaling."""
        # Upscale or ensure consistent sensor resolution (e.g. 260x300 typical for optical platen)
        target_w = max(260, img.width)
        target_h = max(300, img.height)
        img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        arr = np.array(img_resized, dtype=np.float32)

        # Add subtle optical sensor Gaussian noise
        noise = np.random.normal(0.0, 2.5, arr.shape)
        arr = np.clip(arr + noise, 0, 255)

        # Add slight illumination center vignette typical of optical prism lenses
        h, w = arr.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1.0 - (0.05 * (dist_from_center / max_dist))
        arr = np.clip(arr * vignette, 0, 255).astype(np.uint8)

        sensor_img = Image.fromarray(arr, mode="L")
        return sensor_img

    def _synthesize_fallback_ridges(self) -> Image.Image:
        """Generate high-contrast synthetic fingerprint pattern if no local files exist."""
        w, h = 260, 300
        x = np.linspace(-3, 3, w)
        y = np.linspace(-3.5, 3.5, h)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)

        # Spiral ridge pattern modulated by angular harmonic
        ridges = np.sin(r * 22.0 + np.sin(theta * 2.0) * 3.0)
        # Binarize and smooth
        img_arr = ((ridges + 1.0) * 127.5).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L")
        return img.filter(ImageFilter.GaussianBlur(radius=0.8))
