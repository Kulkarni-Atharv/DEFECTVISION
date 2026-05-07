from __future__ import annotations
import cv2
import numpy as np


class ROISelector:
    """Interactive mouse-based ROI selection over a live camera feed."""

    def __init__(self) -> None:
        self.roi: tuple[int, int, int, int] | None = None  # (x, y, w, h)
        self._drawing = False
        self._start: tuple[int, int] | None = None
        self._end: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------
    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = (x, y)
            self._end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            self._end = (x, y)
            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])
            # Require a minimum meaningful area
            if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                self.roi = (x1, y1, x2 - x1, y2 - y1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select(self, cap: cv2.VideoCapture) -> tuple[int, int, int, int] | None:
        """
        Show live feed and let the user drag a rectangle.
        ENTER confirms, R resets, Q aborts.
        Returns (x, y, w, h) or None on abort.
        """
        WIN = "DefectVision — Select Print ROI  [Drag] [ENTER=confirm] [R=reset] [Q=quit]"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self._on_mouse)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            self._draw_guide(display)
            cv2.imshow(WIN, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 and self.roi is not None:   # ENTER
                break
            elif key == ord('r'):
                self.roi = None
                self._start = self._end = None
            elif key == ord('q'):
                cv2.destroyWindow(WIN)
                return None

        cv2.destroyWindow(WIN)
        return self.roi

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _draw_guide(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]

        # In-progress drag
        if self._drawing and self._start and self._end:
            cv2.rectangle(frame, self._start, self._end, (0, 220, 255), 2)

        # Committed ROI
        if self.roi:
            x, y, rw, rh = self.roi
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 80), 2)
            cv2.putText(frame, f"{rw}x{rh} px  — press ENTER to confirm",
                        (x, max(y - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)

        cv2.putText(frame,
                    "Drag to select the print region",
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
