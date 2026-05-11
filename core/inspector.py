from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from config import (
    SSIM_WIN_SIZE,
    PIXEL_DIFF_THRESHOLD,
    SSIM_WEIGHT,
    EDGE_WEIGHT,
    PIXEL_WEIGHT,
    EDGE_SCORE_SCALE,
    PIXEL_SCORE_SCALE,
    DEFECT_SCORE_THRESHOLD,
    ILLUMINATION_CORRECT_ENABLED,
    DEBRIS_DETECTION_ENABLED,
    DEBRIS_MIN_AREA,
    DEBRIS_MAX_AREA,
    DEBRIS_DIFF_THRESHOLD,
    DEBRIS_CIRCULARITY_MIN,
    DEBRIS_MAX_ASPECT_RATIO,
    ADDITION_DETECTION_ENABLED,
    ADDITION_THRESHOLD,
    ADDITION_MIN_AREA,
    ADDITION_BLUR_KSIZE,
    CHAR_INK_CHANGE_THRESHOLD,
)


@dataclass
class CharResult:
    rect: tuple[int, int, int, int]
    ink_change_ratio: float
    is_defect: bool


@dataclass
class InspectionResult:
    # Composite score — informational only; does NOT drive is_defect
    ssim_score: float = 1.0
    edge_diff_score: float = 0.0
    pixel_diff_score: float = 0.0
    defect_score: float = 0.0

    # Primary defect flag — set only by the addition detector
    is_defect: bool = False

    # Per-pixel maps
    ssim_map: np.ndarray | None = None
    diff_map: np.ndarray | None = None
    edge_diff_map: np.ndarray | None = None

    # SSIM-based structural localisation (display only)
    defect_contours: list = field(default_factory=list)
    defect_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    # ---- Primary signal: ink additions in text region ---------------
    # New dark marks (debris, drawn lines, smudges) that were not present
    # in the reference.  This is the ONLY gate that sets is_defect = True.
    addition_contours: list = field(default_factory=list)
    addition_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    addition_count: int = 0

    # Background debris (disabled by default; user cares about text only)
    debris_contours: list = field(default_factory=list)
    debris_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    debris_count: int = 0

    # Per-character analysis
    char_results: list[CharResult] = field(default_factory=list)

    # Diagnostic
    illumination_offset: int = 0


_CANNY_LOW  = 40
_CANNY_HIGH = 120
_MIN_CONTOUR_AREA = 15
_MORPH_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_DEBRIS_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_OPEN_KERNEL   = cv2.getStructuringElement(cv2.MORPH_RECT,    (2, 2))


class Inspector:
    """
    Text-addition inspector.

    The ONLY condition that raises is_defect is:
      New dark marks detected in the text region that were not present in
      the reference (ADDITION_DETECTION_ENABLED).

    Everything else — rotation, translation, lighting drift, slight angle
    change — is handled upstream (FeatureAligner + illumination correction)
    and must NOT trigger a defect on its own.

    SSIM, edge-diff, and pixel-diff are still computed and shown in the
    panel for diagnostic purposes but do not influence is_defect.
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
        if ILLUMINATION_CORRECT_ENABLED:
            live, result.illumination_offset = self._correct_illumination(
                reference, live, text_mask
            )

        # ---- 1. SSIM (informational) ---------------------------------
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

        # ---- 2. Pixel diff (informational) ---------------------------
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

        # ---- 3. Edge diff (informational) ----------------------------
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

        # ---- 4. Composite score (informational) ----------------------
        ssim_component  = float(np.clip(1.0 - ssim_score, 0.0, 1.0))
        edge_component  = float(np.clip(result.edge_diff_score  * EDGE_SCORE_SCALE,  0.0, 1.0))
        pixel_component = float(np.clip(result.pixel_diff_score * PIXEL_SCORE_SCALE, 0.0, 1.0))
        result.defect_score = float(np.clip(
            SSIM_WEIGHT  * ssim_component +
            EDGE_WEIGHT  * edge_component +
            PIXEL_WEIGHT * pixel_component,
            0.0, 1.0,
        ))

        # ---- 5. PRIMARY GATE: addition detection --------------------
        # Detects new dark marks added to the text region.
        # This is the only thing that sets is_defect = True.
        if ADDITION_DETECTION_ENABLED:
            additions = self._detect_additions(reference, live, text_mask)
            result.addition_contours = additions
            result.addition_bboxes   = [cv2.boundingRect(c) for c in additions]
            result.addition_count    = len(additions)
            if result.addition_count > 0:
                result.is_defect    = True
                result.defect_score = max(result.defect_score, DEFECT_SCORE_THRESHOLD)

        # ---- 6. Background debris (optional, off by default) --------
        if DEBRIS_DETECTION_ENABLED:
            debris = self._detect_debris(diff, text_mask)
            result.debris_contours = debris
            result.debris_bboxes   = [cv2.boundingRect(c) for c in debris]
            result.debris_count    = len(debris)

        # ---- 7. SSIM heatmap localisation (display) -----------------
        defect_heat = np.uint8(np.clip((1.0 - ssim_map) * 255, 0, 255))
        _, dmask = cv2.threshold(defect_heat, 50, 255, cv2.THRESH_BINARY)
        dmask = cv2.morphologyEx(dmask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        dmask = cv2.morphologyEx(dmask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
        contours, _ = cv2.findContours(dmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result.defect_contours = [c for c in contours if cv2.contourArea(c) >= _MIN_CONTOUR_AREA]
        result.defect_bboxes   = [cv2.boundingRect(c) for c in result.defect_contours]

        # ---- 8. Per-character ink analysis (display) ----------------
        if text_mask is not None and text_mask.char_rects:
            result.char_results = self._analyse_chars(
                reference, live, diff, text_mask.char_rects
            )

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_additions(
        reference: np.ndarray,
        live: np.ndarray,
        text_mask,
    ) -> list:
        """
        Find new dark marks in the text region that were not in the reference.

        Uses a directional diff: only pixels where live is darker than reference
        are counted.  This makes the detector completely insensitive to:
          - Faded or missing ink (live is lighter → ignored)
          - Lighting increase (whole image brightens → live is lighter → ignored)
          - Rotation / translation after alignment (no net darkness change)

        The search zone is the text mask + a small dilation margin so marks
        drawn right next to a character are also caught.

        A 2×2 MORPH_OPEN removes 1-pixel-wide alignment-edge artifacts before
        blob detection, so a 1-pixel character-boundary shift does not trigger.
        """
        # Auto-detect ink polarity from reference median
        ref_median = float(np.median(reference))
        if ref_median >= 100:
            # Light background, dark ink: added ink makes live darker
            addition_map = np.clip(
                reference.astype(np.int32) - live.astype(np.int32), 0, 255
            ).astype(np.uint8)
        else:
            # Dark background, light ink: added ink makes live brighter
            addition_map = np.clip(
                live.astype(np.int32) - reference.astype(np.int32), 0, 255
            ).astype(np.uint8)

        # Gaussian blur suppresses frame-to-frame sensor noise before thresholding.
        # Real marks (large, strong) survive the blur; single-pixel noise spikes
        # get averaged down below threshold and stop causing per-frame flicker.
        if ADDITION_BLUR_KSIZE > 1:
            ks = ADDITION_BLUR_KSIZE if ADDITION_BLUR_KSIZE % 2 == 1 else ADDITION_BLUR_KSIZE + 1
            addition_map = cv2.GaussianBlur(addition_map, (ks, ks), 0)

        # Restrict to text region + margin; background changes are ignored.
        # Sample the background noise floor BEFORE masking so we can use it
        # to set an adaptive threshold that absorbs residual illumination drift.
        adaptive_thresh = ADDITION_THRESHOLD
        if text_mask is not None and text_mask.is_ready():
            search_zone = cv2.dilate(text_mask.mask, _DEBRIS_KERNEL, iterations=2)
            bg_excl     = cv2.bitwise_not(search_zone)
            bg_vals     = addition_map[bg_excl > 0]
            if len(bg_vals) > 20:
                # 90th-percentile of background "additions" = residual illumination level.
                # Real marks in text are far stronger; we only need to lift the
                # threshold above the noise floor without hiding genuine defects.
                noise_floor = int(np.percentile(bg_vals, 90))
                adaptive_thresh = max(ADDITION_THRESHOLD, noise_floor + 8)
            addition_map = cv2.bitwise_and(addition_map, addition_map, mask=search_zone)

        _, binary = cv2.threshold(addition_map, adaptive_thresh, 255, cv2.THRESH_BINARY)

        # Remove residual 1-pixel-wide alignment artifacts
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _OPEN_KERNEL)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) >= ADDITION_MIN_AREA]

    # ------------------------------------------------------------------
    @staticmethod
    def _correct_illumination(
        reference: np.ndarray,
        live: np.ndarray,
        text_mask,
    ) -> tuple[np.ndarray, int]:
        """
        Affine illumination correction: scale × live + offset.

        Most real lighting changes (lamp intensity, camera exposure) are
        multiplicative — the scene scales proportionally.  A pure additive
        shift cannot compensate for this, leaving a residual that triggers
        false addition detections.  We first apply a scale factor (ratio of
        background medians), then a small residual additive offset to remove
        any remaining constant bias.
        """
        has_mask = text_mask is not None and text_mask.is_ready()
        if has_mask:
            bg_mask   = cv2.bitwise_not(text_mask.mask)
            ref_vals  = reference[bg_mask > 0].astype(np.float32)
            live_vals = live[bg_mask > 0].astype(np.float32)
        else:
            bg_mask   = None
            ref_vals  = reference.ravel().astype(np.float32)
            live_vals = live.ravel().astype(np.float32)

        if len(ref_vals) < 10:
            return live, 0

        ref_med  = float(np.median(ref_vals))
        live_med = float(np.median(live_vals))

        if live_med < 5.0:
            return live, 0

        # Multiplicative correction — handles lamp dimming/brightening, exposure change
        scale = float(np.clip(ref_med / live_med, 0.5, 2.0))
        corrected = np.clip(live.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        # Residual additive offset after scaling
        if bg_mask is not None:
            corr_med = float(np.median(corrected[bg_mask > 0]))
        else:
            corr_med = float(np.median(corrected))
        offset = int(round(ref_med - corr_med))
        if abs(offset) >= 2:
            corrected = np.clip(corrected.astype(np.int32) + offset, 0, 255).astype(np.uint8)

        return corrected, offset

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_debris(diff: np.ndarray, text_mask) -> list:
        """Background-region debris (compact blobs). Off by default."""
        if text_mask is not None and text_mask.is_ready():
            exclusion   = cv2.dilate(text_mask.mask, _DEBRIS_KERNEL, iterations=3)
            search_mask = cv2.bitwise_not(exclusion)
            search_diff = cv2.bitwise_and(diff, diff, mask=search_mask)
        else:
            search_diff = diff

        _, binary = cv2.threshold(search_diff, DEBRIS_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  _DEBRIS_KERNEL)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _DEBRIS_KERNEL)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
