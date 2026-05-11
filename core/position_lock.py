from __future__ import annotations
import cv2
import numpy as np
from config import (
    POSITION_LOCK_THRESHOLD,
    POSITION_LOCK_SEARCH_MARGIN,
    POSITION_LOCK_BLUR_THRESHOLD,
    POSITION_LOCK_ANGLE_RANGE,
    POSITION_LOCK_ANGLE_STEP,
)


class PositionLock:
    """
    Locates the print region in each frame via normalized cross-correlation
    template matching, replacing the fixed-ROI crop used in the static setup.

    Rotation-invariant
    ------------------
    At init (and after each reference recapture) a set of pre-rotated template
    copies is built at every POSITION_LOCK_ANGLE_STEP degrees from
    -POSITION_LOCK_ANGLE_RANGE to +POSITION_LOCK_ANGLE_RANGE.  Each frame all
    rotations are tried and the highest-confidence match wins.  This lets
    PositionLock find the print at any orientation; TextNormalizer then handles
    fine angle correction on the extracted crop.

    Search strategy
    ---------------
    First call (or after losing track): full-frame search — expensive but rare.
    Subsequent calls: restricted ±SEARCH_MARGIN window around last known
    position — completes in < 8 ms on CM5 at 1456×1088 even with 19 templates.

    Gates
    -----
    • Best match confidence < POSITION_LOCK_THRESHOLD  → skip frame (SEARCHING).
    • Laplacian variance of matched crop < POSITION_LOCK_BLUR_THRESHOLD
      → motion-blurred frame, skip.  Set threshold to 0 to disable.
    """

    def __init__(self, template_gray: np.ndarray) -> None:
        self._th, self._tw = template_gray.shape[:2]
        self._margin       = POSITION_LOCK_SEARCH_MARGIN
        self._match_thr    = POSITION_LOCK_THRESHOLD
        self._blur_thr     = POSITION_LOCK_BLUR_THRESHOLD
        self._last_pos: tuple[int, int] | None = None

        self._tpl = template_gray
        self._rotated_templates = self._build_rotation_templates(template_gray)

    # ------------------------------------------------------------------
    def set_template(self, template_gray: np.ndarray) -> None:
        """
        Replace the reference template and rebuild all rotated copies.
        Call this instead of directly setting _tpl when recapturing reference.
        """
        self._tpl = template_gray
        self._th, self._tw = template_gray.shape[:2]
        self._rotated_templates = self._build_rotation_templates(template_gray)
        self._last_pos = None

    def reset(self) -> None:
        """Force a full-frame search on the next find() call."""
        self._last_pos = None

    # ------------------------------------------------------------------
    @staticmethod
    def _build_rotation_templates(tpl: np.ndarray) -> list[np.ndarray]:
        """
        Pre-rotate the template at every ANGLE_STEP from -RANGE to +RANGE.
        Using BORDER_REPLICATE so edge pixels don't become black and distort
        the NCC score.
        """
        h, w  = tpl.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        templates: list[np.ndarray] = []

        angle = -POSITION_LOCK_ANGLE_RANGE
        while angle <= POSITION_LOCK_ANGLE_RANGE + 0.5:
            if abs(angle) < 0.5:
                templates.append(tpl)
            else:
                M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
                rotated = cv2.warpAffine(
                    tpl, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                templates.append(rotated)
            angle += POSITION_LOCK_ANGLE_STEP

        return templates

    # ------------------------------------------------------------------
    def find(
        self, frame_gray: np.ndarray
    ) -> tuple[tuple[int, int, int, int], float] | None:
        """
        Search for the template at all pre-rotated angles in *frame_gray*.

        Returns
        -------
        ((x, y, w, h), confidence)  in full-frame pixel coordinates, or
        None if the print is not found or the frame is too blurry.
        """
        fh, fw = frame_gray.shape

        # --- choose search region ----------------------------------------
        if self._last_pos is not None:
            lx, ly = self._last_pos
            x1 = max(0, lx - self._margin)
            y1 = max(0, ly - self._margin)
            x2 = min(fw, lx + self._tw + self._margin)
            y2 = min(fh, ly + self._th + self._margin)
            region = frame_gray[y1:y2, x1:x2]
            ox, oy = x1, y1
        else:
            region = frame_gray
            ox, oy = 0, 0

        if region.shape[0] < self._th or region.shape[1] < self._tw:
            self._last_pos = None
            return None

        # --- try all rotations, keep best --------------------------------
        best_conf = -1.0
        best_mx   = 0
        best_my   = 0

        for tpl in self._rotated_templates:
            result = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
            _, conf, _, loc = cv2.minMaxLoc(result)
            if conf > best_conf:
                best_conf = conf
                best_mx   = loc[0] + ox
                best_my   = loc[1] + oy

        if best_conf < self._match_thr:
            self._last_pos = None
            return None

        # Clamp to frame bounds
        best_mx = int(np.clip(best_mx, 0, fw - self._tw))
        best_my = int(np.clip(best_my, 0, fh - self._th))

        # --- blur gate ---------------------------------------------------
        if self._blur_thr > 0:
            crop    = frame_gray[best_my: best_my + self._th, best_mx: best_mx + self._tw]
            lap_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            if lap_var < self._blur_thr:
                return None   # motion-blurred; don't update last_pos

        self._last_pos = (best_mx, best_my)
        return (best_mx, best_my, self._tw, self._th), float(best_conf)
