from __future__ import annotations
import cv2
import numpy as np
from config import TEXT_NORM_ENABLED, TEXT_NORM_MIN_ANGLE


class TextNormalizer:
    """
    Makes the inspection pipeline angle-invariant for rotating objects.

    Problem
    -------
    A printing roller presents the text at a slightly different in-plane
    rotation each pass (depending on exact frame timing vs. RPM).  Pixel-
    based comparison fails unless both reference and live are at the same
    orientation.

    Solution
    --------
    Estimate the dominant text orientation by running PCA on the ink-pixel
    coordinates.  The principal axis of an ink-pixel cloud equals the text
    line direction regardless of capture angle.

    At reference capture:  store the reference ink angle (ref_angle).
    Each live frame:        compute live ink angle, rotate live by
                            (ref_angle − live_angle) → live is now at the
                            same orientation as the reference.

    The FeatureAligner then handles any remaining small translation/rotation
    (< 3°) caused by sub-pixel jitter.

    Performance
    -----------
    ~3 ms per frame on a Raspberry Pi CM5 (binarize + PCA + warpAffine).

    Edge cases
    ----------
    • Fewer than MIN_INK_PX ink pixels after erosion → angle = 0, no rotation.
    • Eigenvalue ratio close to 1 (isotropic blob) → angle unreliable, skip.
    • Angle delta < TEXT_NORM_MIN_ANGLE degrees → skip rotation (negligible).
    """

    MIN_INK_PX       = 30     # need at least this many ink pixels for reliable PCA
    ISO_RATIO_THRESH = 0.70   # if λ2/λ1 > this the distribution is too isotropic

    def __init__(self) -> None:
        self._ref_angle: float = 0.0

    # ------------------------------------------------------------------
    def set_reference(self, ref_gray: np.ndarray) -> None:
        """Compute and cache the reference ink angle."""
        self._ref_angle = self._ink_angle(ref_gray)

    def normalize(self, live_gray: np.ndarray) -> np.ndarray:
        """
        Rotate live_gray so its text is at the same angle as the reference.
        Returns live_gray unchanged if normalization is disabled or the angle
        difference is below TEXT_NORM_MIN_ANGLE.
        """
        if not TEXT_NORM_ENABLED:
            return live_gray

        live_angle = self._ink_angle(live_gray)
        delta = self._ref_angle - live_angle

        # Wrap to [-180, 180]
        delta = (delta + 180.0) % 360.0 - 180.0

        if abs(delta) < TEXT_NORM_MIN_ANGLE:
            return live_gray

        return self._rotate(live_gray, delta)

    # ------------------------------------------------------------------
    @classmethod
    def _ink_angle(cls, gray: np.ndarray) -> float:
        """
        PCA on ink-pixel coordinates → principal axis angle (degrees).

        The principal axis is the direction along which the ink pixels
        have maximum spread.  For a line of text this equals the text
        baseline direction, independent of the capture angle.
        """
        # Binarize: auto-detect dark-on-light vs light-on-dark
        median_val = float(np.median(gray))
        thresh_type = (cv2.THRESH_BINARY_INV if median_val >= 100
                       else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
        _, binary = cv2.threshold(gray, 0, 255, thresh_type)

        # Light erosion removes single-pixel sensor noise from the point cloud
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.erode(binary, kernel, iterations=1)

        y_pts, x_pts = np.where(binary > 0)
        if len(x_pts) < cls.MIN_INK_PX:
            return 0.0

        pts = np.column_stack([x_pts, y_pts]).astype(np.float32)
        _, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

        # If the distribution is nearly isotropic (round blob, no clear axis),
        # the principal axis is unreliable — return 0 to skip rotation.
        if eigenvalues[1, 0] / max(eigenvalues[0, 0], 1e-6) > cls.ISO_RATIO_THRESH:
            return 0.0

        vx, vy = float(eigenvectors[0, 0]), float(eigenvectors[0, 1])
        angle = float(np.degrees(np.arctan2(vy, vx)))

        # Fold into [-90, 90]: a line at +100° is the same text as at -80°
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0

        return angle

    @staticmethod
    def _rotate(gray: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
