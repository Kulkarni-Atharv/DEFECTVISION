from __future__ import annotations
import cv2
import numpy as np


class ROISelector:
    """
    ROI selection using cv2.selectROI — OpenCV's built-in selector.
    Handles its own window and mouse events internally, avoiding the
    setMouseCallback / NULL-window-handler crash on Pi OS Qt builds.

    Usage: drag a rectangle, press SPACE or ENTER to confirm, C to cancel.
    """

    def select(self, cam) -> tuple[int, int, int, int] | None:
        """
        Capture a live frame, freeze it, and let the user drag the ROI.
        Returns (x, y, w, h) or None if cancelled.
        """
        # Flush stale frames from the camera buffer
        for _ in range(5):
            cam.read()

        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Could not read frame for ROI selection.")
            return None

        WIN = "Select ROI — Drag with mouse | SPACE or ENTER = confirm | C = cancel"

        r = cv2.selectROI(WIN, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(WIN)

        # selectROI returns (0, 0, 0, 0) when cancelled with C
        if r[2] < 16 or r[3] < 16:
            return None

        x, y, w, h = (int(v) for v in r)
        print(f"[INFO] ROI selected: x={x} y={y} w={w} h={h}")
        return (x, y, w, h)
