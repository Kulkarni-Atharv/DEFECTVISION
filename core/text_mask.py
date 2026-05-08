from __future__ import annotations
import cv2
import numpy as np
from config import TEXT_MASK_DILATE_ITER, TEXT_MASK_MIN_CHAR_AREA


class TextMask:
    """
    Extracts a binary mask of ink/text pixels from the reference image.

    Purpose
    -------
    Focuses the inspection on character pixels only.  Pixel-diff and
    edge-diff scores are then normalised against the number of TEXT pixels
    rather than the total ROI area, so a defect affecting 10 % of ink is
    scored as 0.10 — not diluted by a large background.

    Algorithm
    ---------
    1. Otsu threshold (auto dark-on-light / light-on-dark detection).
    2. Morphological dilation by TEXT_MASK_DILATE_ITER to include edge
       anti-alias pixels that sit just outside the raw binary boundary.
    3. Connected-component analysis to identify individual character blobs
       (used for per-character ink-density checks in the Inspector).
    """

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None
        self._char_rects: list[tuple[int, int, int, int]] = []
        self._pixel_count: int = 0
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ------------------------------------------------------------------
    def build(self, ref_gray: np.ndarray) -> None:
        """Compute mask and character bounding boxes from the reference."""
        # Auto-detect polarity: dark text on light background is most common;
        # if median is dark (< 127) we have light text on dark background.
        median_val = float(np.median(ref_gray))
        if median_val >= 100:
            thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU   # dark ink
        else:
            thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU        # light ink

        _, binary = cv2.threshold(ref_gray, 0, 255, thresh_type)

        # Dilate to capture anti-aliased edge pixels
        mask = cv2.dilate(
            binary, self._dilate_kernel, iterations=TEXT_MASK_DILATE_ITER
        )
        self._mask = mask
        self._pixel_count = int(np.count_nonzero(mask))

        # Character-level bounding boxes via connected components on
        # the un-dilated binary (cleaner character separation)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        self._char_rects = []
        for i in range(1, n_labels):        # skip background label 0
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= TEXT_MASK_MIN_CHAR_AREA:
                self._char_rects.append((
                    int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_WIDTH]),
                    int(stats[i, cv2.CC_STAT_HEIGHT]),
                ))

    # ------------------------------------------------------------------
    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    @property
    def char_rects(self) -> list[tuple[int, int, int, int]]:
        return self._char_rects

    @property
    def pixel_count(self) -> int:
        """Number of foreground (ink) pixels in the mask."""
        return self._pixel_count

    def is_ready(self) -> bool:
        return self._mask is not None and self._pixel_count > 0
