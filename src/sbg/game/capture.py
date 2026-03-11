"""Screen capture using dxcam (primary) with mss fallback."""

import numpy as np


class ScreenCapture:
    """Captures game frames from screen using DirectX (dxcam) or mss fallback."""

    def __init__(self, monitor: int = 0, region: tuple | None = None, fps: int = 30):
        self.monitor = monitor
        self.region = tuple(region) if region else None
        self.fps = fps
        self._camera = None
        self._use_dxcam = True

    def start(self):
        try:
            import dxcam

            self._camera = dxcam.create(device_idx=self.monitor)
            # dxcam expects (left, top, right, bottom) but our region is
            # (left, top, width, height) — convert here
            dxcam_region = None
            if self.region:
                l, t, w, h = self.region
                dxcam_region = (l, t, l + w, t + h)
            self._camera.start(
                target_fps=self.fps,
                region=dxcam_region,
            )
        except Exception:
            self._use_dxcam = False
            import mss

            self._camera = mss.mss()

    def grab(self) -> np.ndarray:
        """Grab a single frame as a numpy array (H, W, C) in BGR format."""
        if self._use_dxcam:
            frame = self._camera.get_latest_frame()
            if frame is None:
                # dxcam can return None if no new frame is ready
                import time
                time.sleep(0.001)
                frame = self._camera.get_latest_frame()
            return frame  # Already numpy, RGB format
        else:
            import cv2

            monitor = self._camera.monitors[self.monitor + 1]
            if self.region:
                monitor = {
                    "left": self.region[0],
                    "top": self.region[1],
                    "width": self.region[2],
                    "height": self.region[3],
                }
            shot = self._camera.grab(monitor)
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2RGB)

    def stop(self):
        if self._use_dxcam and self._camera:
            self._camera.stop()
        elif not self._use_dxcam and self._camera:
            self._camera.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
