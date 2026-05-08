from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from config import (
    SSIM_THRESHOLD,
    SSIM_WIN_SIZE,
    EDGE_DIFF_THRESHOLD,
    PIXEL_DIFF_THRESHOLD,
    SSIM_WEIGHT,
    EDGE_WEIGHT,
    PIXEL_WEIGHT,
    EDGE_SCORE_SCALE,
    PIXEL_SCORE_SCALE,
    DEFECT_SCORE_THRESHOLD,
    CHANGED_PIXEL_RATIO_THRESHOLD,
    CHAR_INK_CHANGE_THRESHOLD,
)


@dataclass
class CharResult:
    rect: tuple[int, int, int, int]   # (x, y, w, h) in ROI coords
    ink_change_ratio: float           # 0 = identical, 1 = completely different
    is_defect: bool


@dataclass
class InspectionResult:
    # Scalar metrics
    ssim_score: float = 1.0
    edge_diff_score: float = 0.0
    pixel_diff_score: float = 0.0
    defect_score: float = 0.0
    is_defect: bool = False

    # Per-pixel maps (same spatial size as ROI)
    ssim_map: np.ndarray | None = None        # float64 [−1, 1]
    diff_map: np.ndarray | None = None        # uint8 absolute diff
    edge_diff_map: np.ndarray | None = None   # uint8 binary edge XOR

    # Localisation — whole-ROI contours
    defect_contours: list = field(default_factory=list)
    defect_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    # Character-level results (populated when TextMask is supplied)
    char_results: list[CharResult] = field(default_factory=list)


# Canny tuned for small industrial text
_CANNY_LOW  = 40
_CANNY_HIGH = 120
_MIN_CONTOUR_AREA = 15
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class Inspector:
    """
    Core inspection engine — text-aware mode.

    When a TextMask is supplied (recommended), pixel-diff and edge-diff
    scores are normalised against ink-pixel count rather than total ROI
    area.  This means a defect on 10 % of the characters scores ~0.10
    regardless of how large the background is.

    Character-level analysis checks each connected-component blob for
    ink-density changes (fading, missing strokes) independently.

    Three complementary signals → composite defect score:
      SSIM        — structural / luminance / contrast (full ROI)
      Edge diff   — broken/missing strokes (text region only)
      Pixel diff  — raw intensity change  (text region only)
    """

    def __init__(self) -> None:
        self._ref_edges: np.ndarray | None = None

    def set_reference(self, ref_gray: np.ndarray) -> None:
        self._ref_edges = self._edges(ref_gray)

    # ------------------------------------------------------------------
    def inspect(
        self,
        reference: np.ndarray,
        live: np.ndarray,
        text_mask=None,          # core.text_mask.TextMask | None
    ) -> InspectionResult:
        result = InspectionResult()

        if reference.shape != live.shape:
            live = cv2.resize(live, (reference.shape[1], reference.shape[0]))

        # ---- 1. SSIM (full ROI) -------------------------------------
        win = min(SSIM_WIN_SIZE, reference.shape[0], reference.shape[1])
        win = win if win % 2 == 1 else win - 1
        win = max(win, 3)

        ssim_score, ssim_map = ssim_fn(
            reference, live,
            full=True, data_range=255,
            win_size=win, gaussian_weights=True,
        )
        result.ssim_score = float(ssim_score)
        result.ssim_map   = ssim_map

        # ---- 2. Pixel difference ------------------------------------
        diff = cv2.absdiff(reference, live)
        result.diff_map = diff

        if text_mask is not None and text_mask.is_ready():
            mask        = text_mask.mask
            norm_denom  = float(text_mask.pixel_count)
            text_diff   = cv2.bitwise_and(diff, diff, mask=mask)
            result.pixel_diff_score = float(
                np.count_nonzero(text_diff > PIXEL_DIFF_THRESHOLD) / norm_denom
            )
        else:
            norm_denom = float(diff.size)
            result.pixel_diff_score = float(
                np.count_nonzero(diff > PIXEL_DIFF_THRESHOLD) / norm_denom
            )

        # ---- 3. Edge difference -------------------------------------
        live_edges = self._edges(live)
        if self._ref_edges is None:
            self._ref_edges = self._edges(reference)
        edge_diff = cv2.bitwise_xor(self._ref_edges, live_edges)
        result.edge_diff_map = edge_diff

        if text_mask is not None and text_mask.is_ready():
            text_edge = cv2.bitwise_and(edge_diff, edge_diff, mask=text_mask.mask)
            result.edge_diff_score = float(
                np.count_nonzero(text_edge) / float(text_mask.pixel_count)
            )
        else:
            result.edge_diff_score = float(
                np.count_nonzero(edge_diff) / float(edge_diff.size)
            )

        # ---- 4. Composite defect score ------------------------------
        ssim_component  = float(np.clip(1.0 - ssim_score, 0.0, 1.0))
        edge_component  = float(np.clip(result.edge_diff_score  * EDGE_SCORE_SCALE,  0.0, 1.0))
        pixel_component = float(np.clip(result.pixel_diff_score * PIXEL_SCORE_SCALE, 0.0, 1.0))

        defect_score = (
            SSIM_WEIGHT  * ssim_component +
            EDGE_WEIGHT  * edge_component +
            PIXEL_WEIGHT * pixel_component
        )
        result.defect_score = float(np.clip(defect_score, 0.0, 1.0))
        result.is_defect    = result.defect_score >= DEFECT_SCORE_THRESHOLD

        # ---- Hard pixel-change override -----------------------------
        if (CHANGED_PIXEL_RATIO_THRESHOLD > 0
                and result.pixel_diff_score >= CHANGED_PIXEL_RATIO_THRESHOLD):
            result.is_defect    = True
            result.defect_score = max(result.defect_score, DEFECT_SCORE_THRESHOLD)

        # ---- 5. Defect localisation (SSIM heatmap contours) --------
        defect_heat = np.uint8(np.clip((1.0 - ssim_map) * 255, 0, 255))
        _, defect_mask_img = cv2.threshold(defect_heat, 50, 255, cv2.THRESH_BINARY)
        defect_mask_img = cv2.morphologyEx(defect_mask_img, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        defect_mask_img = cv2.morphologyEx(defect_mask_img, cv2.MORPH_OPEN,  _MORPH_KERNEL)

        contours, _ = cv2.findContours(
            defect_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result.defect_contours = [c for c in contours if cv2.contourArea(c) >= _MIN_CONTOUR_AREA]
        result.defect_bboxes   = [cv2.boundingRect(c) for c in result.defect_contours]

        # ---- 6. Character-level ink analysis ------------------------
        if text_mask is not None and text_mask.char_rects:
            result.char_results = self._analyse_chars(
                reference, live, diff, text_mask.char_rects
            )

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _analyse_chars(
        reference: np.ndarray,
        live: np.ndarray,
        diff: np.ndarray,
        char_rects: list[tuple[int, int, int, int]],
    ) -> list[CharResult]:
        results = []
        for rect in char_rects:
            x, y, w, h = rect
            ref_crop  = reference[y:y + h, x:x + w]
            live_crop = live[y:y + h, x:x + w]
            diff_crop = diff[y:y + h, x:x + w]

            ref_mean  = float(np.mean(ref_crop))  + 1e-6
            live_mean = float(np.mean(live_crop))
            # Normalised absolute ink-density change
            ink_change = abs(live_mean - ref_mean) / ref_mean

            # Also check what fraction of the character's pixels changed
            changed_frac = float(np.count_nonzero(diff_crop > PIXEL_DIFF_THRESHOLD)) / max(diff_crop.size, 1)

            score    = max(ink_change, changed_frac)
            is_def   = score >= CHAR_INK_CHANGE_THRESHOLD
            results.append(CharResult(rect=rect, ink_change_ratio=score, is_defect=is_def))
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _edges(img: np.ndarray) -> np.ndarray:
        return cv2.Canny(img, _CANNY_LOW, _CANNY_HIGH)
