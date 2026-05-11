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
    ILLUMINATION_CORRECT_ENABLED,
    DEBRIS_DETECTION_ENABLED,
    DEBRIS_MIN_AREA,
    DEBRIS_MAX_AREA,
    DEBRIS_DIFF_THRESHOLD,
    DEBRIS_CIRCULARITY_MIN,
    DEBRIS_MAX_ASPECT_RATIO,
)


@dataclass
class CharResult:
    rect: tuple[int, int, int, int]
    ink_change_ratio: float
    is_defect: bool


@dataclass
class InspectionResult:
    # Scalar metrics
    ssim_score: float = 1.0
    edge_diff_score: float = 0.0
    pixel_diff_score: float = 0.0
    defect_score: float = 0.0
    is_defect: bool = False

    # Per-pixel maps
    ssim_map: np.ndarray | None = None
    diff_map: np.ndarray | None = None
    edge_diff_map: np.ndarray | None = None

    # Localisation — structural defects from SSIM
    defect_contours: list = field(default_factory=list)
    defect_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    # Debris blobs (compact foreign objects, illumination-corrected)
    debris_contours: list = field(default_factory=list)
    debris_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    debris_count: int = 0

    # Character-level results
    char_results: list[CharResult] = field(default_factory=list)

    # Diagnostic
    illumination_offset: int = 0   # brightness correction applied to live frame


# Canny tuned for small industrial text
_CANNY_LOW  = 40
_CANNY_HIGH = 120
_MIN_CONTOUR_AREA = 15
_MORPH_KERNEL    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_DEBRIS_KERNEL   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


class Inspector:
    """
    Text-aware inspection engine with illumination correction.

    Pipeline per frame
    ------------------
    0. Illumination correction  — remove global brightness drift using the
       background (non-ink) region as a reference.  Eliminates false positives
       from lighting variation before any difference is computed.

    1. SSIM                     — structural similarity over the full ROI.
                                  Catches blurry/faded characters and contrast changes.

    2. Pixel diff (text-masked) — absolute intensity diff, normalised against
                                  ink-pixel count.  Sensitive to ink changes only.

    3. Edge diff (text-masked)  — Canny XOR, normalised against ink-pixel count.
                                  Catches missing/broken strokes.

    4. Composite score          — weighted combination → is_defect flag.

    5. Debris detection         — compact blob search in the illumination-corrected
                                  diff image.  Shape filter (circularity + aspect
                                  ratio) rejects elongated alignment noise and keeps
                                  only foreign-object signatures.  A single debris
                                  blob overrides is_defect regardless of SSIM score.

    6. SSIM heatmap localisation
    7. Per-character ink analysis
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
        text_mask=None,
    ) -> InspectionResult:
        result = InspectionResult()

        if reference.shape != live.shape:
            live = cv2.resize(live, (reference.shape[1], reference.shape[0]))

        # ---- 0. Illumination correction ------------------------------
        # Measure how much the background region has shifted in brightness
        # and add that offset back to the live frame so the diff reflects
        # only genuine content changes, not lighting drift.
        if ILLUMINATION_CORRECT_ENABLED:
            live, result.illumination_offset = self._correct_illumination(
                reference, live, text_mask
            )

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

        # ---- 2. Pixel difference (text region) ----------------------
        diff = cv2.absdiff(reference, live)
        result.diff_map = diff

        if text_mask is not None and text_mask.is_ready():
            text_diff = cv2.bitwise_and(diff, diff, mask=text_mask.mask)
            norm_denom = float(text_mask.pixel_count)
            result.pixel_diff_score = float(
                np.count_nonzero(text_diff > PIXEL_DIFF_THRESHOLD) / norm_denom
            )
        else:
            result.pixel_diff_score = float(
                np.count_nonzero(diff > PIXEL_DIFF_THRESHOLD) / float(diff.size)
            )

        # ---- 3. Edge difference (text region) -----------------------
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

        # Hard pixel-change override
        if (CHANGED_PIXEL_RATIO_THRESHOLD > 0
                and result.pixel_diff_score >= CHANGED_PIXEL_RATIO_THRESHOLD):
            result.is_defect    = True
            result.defect_score = max(result.defect_score, DEFECT_SCORE_THRESHOLD)

        # ---- 5. Debris detection ------------------------------------
        # Runs on the illumination-corrected diff — lighting drift has already
        # been removed, so any compact bright blob in the diff is a real
        # foreign object, not a brightness ghost.
        if DEBRIS_DETECTION_ENABLED:
            debris = self._detect_debris(diff, text_mask)
            result.debris_contours = debris
            result.debris_bboxes   = [cv2.boundingRect(c) for c in debris]
            result.debris_count    = len(debris)
            if result.debris_count > 0:
                # Debris is always a defect regardless of composite score
                result.is_defect    = True
                result.defect_score = max(result.defect_score, DEFECT_SCORE_THRESHOLD)

        # ---- 6. Defect localisation (SSIM heatmap) ------------------
        defect_heat = np.uint8(np.clip((1.0 - ssim_map) * 255, 0, 255))
        _, defect_mask_img = cv2.threshold(defect_heat, 50, 255, cv2.THRESH_BINARY)
        defect_mask_img = cv2.morphologyEx(defect_mask_img, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        defect_mask_img = cv2.morphologyEx(defect_mask_img, cv2.MORPH_OPEN,  _MORPH_KERNEL)

        contours, _ = cv2.findContours(
            defect_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result.defect_contours = [c for c in contours if cv2.contourArea(c) >= _MIN_CONTOUR_AREA]
        result.defect_bboxes   = [cv2.boundingRect(c) for c in result.defect_contours]

        # ---- 7. Character-level ink analysis ------------------------
        if text_mask is not None and text_mask.char_rects:
            result.char_results = self._analyse_chars(
                reference, live, diff, text_mask.char_rects
            )

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _correct_illumination(
        reference: np.ndarray,
        live: np.ndarray,
        text_mask,
    ) -> tuple[np.ndarray, int]:
        """
        Estimate global brightness drift from background pixels and apply
        an additive correction to live so reference and live have the same
        median background intensity.

        Background pixels are the best illumination reference because they
        should be constant between frames — any change there is purely
        due to lighting, not content.
        """
        if text_mask is not None and text_mask.is_ready():
            bg_mask   = cv2.bitwise_not(text_mask.mask)
            ref_vals  = reference[bg_mask > 0].astype(np.float32)
            live_vals = live[bg_mask > 0].astype(np.float32)
        else:
            ref_vals  = reference.ravel().astype(np.float32)
            live_vals = live.ravel().astype(np.float32)

        if len(ref_vals) == 0:
            return live, 0

        offset = int(round(float(np.median(ref_vals)) - float(np.median(live_vals))))
        if abs(offset) < 2:         # negligible drift
            return live, 0

        corrected = np.clip(live.astype(np.int32) + offset, 0, 255).astype(np.uint8)
        return corrected, offset

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_debris(
        diff: np.ndarray,
        text_mask,
    ) -> list:
        """
        Find compact foreign-object blobs in the illumination-corrected diff.

        Works across both background AND text regions so debris resting ON
        a character is also caught.

        Shape filters
        -------------
        circularity  = 4π·area / perimeter²   (1 = perfect circle)
        aspect_ratio = max(w,h) / min(w,h)     (1 = square)

        Debris is compact (high circularity OR low aspect ratio).
        Alignment-noise artifacts are elongated — they fail both filters.
        """
        # Binary threshold on the diff
        _, binary = cv2.threshold(diff, DEBRIS_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Remove single-pixel noise; fill small holes in blobs
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  _DEBRIS_KERNEL)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _DEBRIS_KERNEL)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        debris = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < DEBRIS_MIN_AREA or area > DEBRIS_MAX_AREA:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter < 1.0:
                continue

            circularity  = 4.0 * np.pi * area / (perimeter * perimeter)
            _, _, bw, bh = cv2.boundingRect(c)
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)

            # Accept if compact by either metric
            if circularity >= DEBRIS_CIRCULARITY_MIN or aspect_ratio <= DEBRIS_MAX_ASPECT_RATIO:
                debris.append(c)

        return debris

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

            ref_mean   = float(np.mean(ref_crop)) + 1e-6
            live_mean  = float(np.mean(live_crop))
            ink_change = abs(live_mean - ref_mean) / ref_mean

            changed_frac = float(
                np.count_nonzero(diff_crop > PIXEL_DIFF_THRESHOLD)
            ) / max(diff_crop.size, 1)

            score  = max(ink_change, changed_frac)
            is_def = score >= CHAR_INK_CHANGE_THRESHOLD
            results.append(CharResult(rect=rect, ink_change_ratio=score, is_defect=is_def))
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _edges(img: np.ndarray) -> np.ndarray:
        return cv2.Canny(img, _CANNY_LOW, _CANNY_HIGH)
