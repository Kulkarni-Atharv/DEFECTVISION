from __future__ import annotations
import cv2
import numpy as np
from config import (
    FEATURE_ALIGNER_ENABLED,
    ORB_N_FEATURES,
    MIN_MATCH_COUNT,
    HOMOGRAPHY_RANSAC_THRESH,
    ALIGN_MAX_SHIFT_RATIO,
)


class FeatureAligner:
    """
    Rotation-invariant alignment using ORB keypoints + homography.

    Replaces the phase-correlation Aligner for crops that may have
    small rotational variance (e.g. bottle label rotation on a roller).

    Strategy
    --------
    1. Detect ORB keypoints in reference once (set_reference).
    2. Per frame: detect keypoints in live crop, BF-match, compute
       homography via RANSAC, warp live to reference geometry.
    3. Fallback: if fewer than MIN_MATCH_COUNT inliers survive RANSAC,
       fall back to phase-correlation translation-only alignment.  This
       handles very small or texture-poor ROIs where ORB finds no features.

    Returns (aligned_gray, confidence) where confidence is the RANSAC
    inlier ratio (0–1).  A confidence of -1.0 signals that the phase-
    correlation fallback was used.
    """

    def __init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES, fastThreshold=7)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._ref_kp   = None
        self._ref_des  = None
        self._ref_shape: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    def set_reference(self, ref_gray: np.ndarray) -> None:
        kp, des = self._orb.detectAndCompute(ref_gray, None)
        self._ref_kp    = kp
        self._ref_des   = des
        self._ref_shape = ref_gray.shape[:2]   # (h, w)

    # ------------------------------------------------------------------
    def align(
        self, ref_gray: np.ndarray, live_gray: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Warp live_gray to align with ref_gray.

        Returns
        -------
        (aligned, confidence)
          confidence = RANSAC inlier ratio   [0, 1]
                     = -1.0 if phase-correlation fallback was used
                     =  0.0 if no alignment was possible
        """
        if not FEATURE_ALIGNER_ENABLED:
            return live_gray, 1.0

        if self._ref_des is None:
            self.set_reference(ref_gray)

        h, w = ref_gray.shape[:2]
        if live_gray.shape[:2] != (h, w):
            live_gray = cv2.resize(live_gray, (w, h))

        live_kp, live_des = self._orb.detectAndCompute(live_gray, None)

        # ---- ORB homography path ------------------------------------
        if (live_des is not None
                and self._ref_des is not None
                and len(live_kp) >= MIN_MATCH_COUNT
                and len(self._ref_kp) >= MIN_MATCH_COUNT):

            matches = self._matcher.match(self._ref_des, live_des)
            matches = sorted(matches, key=lambda m: m.distance)
            good = matches[:min(len(matches), 60)]

            if len(good) >= MIN_MATCH_COUNT:
                src_pts = np.float32(
                    [self._ref_kp[m.queryIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [live_kp[m.trainIdx].pt for m in good]
                ).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(
                    dst_pts, src_pts,
                    cv2.RANSAC,
                    HOMOGRAPHY_RANSAC_THRESH,
                )
                if H is not None and mask is not None:
                    inliers = int(mask.ravel().sum())
                    if inliers >= MIN_MATCH_COUNT:
                        aligned = cv2.warpPerspective(
                            live_gray, H, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
                        return aligned, float(inliers / len(mask))

        # ---- Phase-correlation fallback (translation only) ----------
        try:
            shift, _ = cv2.phaseCorrelate(
                np.float32(ref_gray), np.float32(live_gray)
            )
            dx, dy = shift
            max_dx = w * ALIGN_MAX_SHIFT_RATIO
            max_dy = h * ALIGN_MAX_SHIFT_RATIO
            if abs(dx) <= max_dx and abs(dy) <= max_dy:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned = cv2.warpAffine(
                    live_gray, M, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                return aligned, -1.0
        except cv2.error:
            pass

        return live_gray, 0.0
