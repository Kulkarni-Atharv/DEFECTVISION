# ============================================================
# DefectVision — Print Defect Inspection System
# All tunable parameters live here.
# ============================================================

# ---- Camera -------------------------------------------------
# Backend options:
#   "PICAMERA2"  → picamera2 (Raspberry Pi CM5 / Pi 5)
#   "DSHOW"      → cv2.VideoCapture with DirectShow (Windows)
#   "V4L2"       → cv2.VideoCapture with V4L2 (Linux, non-Pi)
#   "AUTO"       → cv2.VideoCapture with auto backend
CAMERA_BACKEND = "PICAMERA2"

# OpenCV backend settings (used when CAMERA_BACKEND != "PICAMERA2")
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Picamera2 backend settings (used when CAMERA_BACKEND = "PICAMERA2")
PICAMERA2_WIDTH    = 1456   # IMX296 native width
PICAMERA2_HEIGHT   = 1088   # IMX296 native height
PICAMERA2_FPS      = 30
PICAMERA2_WARMUP_S = 2.0    # AEC/AWB settle time (seconds)

# ---- Reference capture --------------------------------------
REFERENCE_FRAME_COUNT = 10   # Frames averaged to build the clean reference
REFERENCE_WARMUP_FRAMES = 10 # Discard this many frames before capturing (sensor warm-up)

# ---- Preprocessing ------------------------------------------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
DENOISE_KERNEL_SIZE = 3      # Gaussian blur kernel size (must be odd); 1 = disabled

# ---- Alignment (phase correlation) --------------------------
ALIGN_ENABLED = True
# Max fraction of ROI size allowed as shift — rejects unreliable correlations.
# With position lock active, template-match variance can be 10-15 px, so this
# must be large enough for the aligner to cover that residual offset.
ALIGN_MAX_SHIFT_RATIO = 0.25  # 25 % of ROI width/height (was 0.08)

# ---- Inspection thresholds ----------------------------------
# SSIM: 0 = completely different, 1 = identical
SSIM_THRESHOLD = 0.80
SSIM_WIN_SIZE = 5            # Smaller window = catches finer / more localised defects

# Edge difference: fraction of pixels whose edges differ
EDGE_DIFF_THRESHOLD = 0.06

# Raw pixel difference: absolute intensity difference per pixel (0–255)
PIXEL_DIFF_THRESHOLD = 15    # Lowered from 30 — catches subtle debris and fine strings

# ---- Defect scoring (weighted combination) ------------------
SSIM_WEIGHT   = 0.50
EDGE_WEIGHT   = 0.25
PIXEL_WEIGHT  = 0.25

# Scaling factors that normalise edge/pixel fraction scores into [0, 1].
# Lower values = more tolerant of alignment-induced edge halos on moving objects.
# Raise them back toward 6.0 / 12.0 for a static-camera setup.
EDGE_SCORE_SCALE  = 4.0   # applied to edge_diff_score  (was hardcoded 6.0)
PIXEL_SCORE_SCALE = 8.0   # applied to pixel_diff_score (was hardcoded 12.0)

# Combined defect score: 0.0 = perfect, 1.0 = severe defect.
# Raised from 0.18 — moving-object alignment noise raises the baseline score.
DEFECT_SCORE_THRESHOLD = 0.32

# ---- Temporal consistency filter ----------------------------
# A defect is confirmed only when TEMPORAL_DEFECT_RATIO of the last
# TEMPORAL_WINDOW frames independently flag a defect.
# Wider window + higher ratio = more stable output, slower response.
TEMPORAL_WINDOW       = 10   # frames in the rolling window
TEMPORAL_DEFECT_RATIO = 0.70 # 70 % of frames must agree before confirming

# ---- Visualization ------------------------------------------
HEATMAP_ALPHA = 0.45         # 0 = no heatmap, 1 = full heatmap overlay
ROI_BORDER_THICKNESS = 3
PANEL_CELL_SCALE = 2.5       # Display scale multiplier for each panel cell
CORNER_ACCENT_LENGTH = 18    # Length of corner bracket lines on main feed

# ---- Position Lock (moving object tracking) -------------------------
# Replaces the fixed-ROI crop with template matching so the print region
# is found dynamically each frame, regardless of conveyor position.
POSITION_LOCK_ENABLED        = True
POSITION_LOCK_THRESHOLD      = 0.55   # Lowered from 0.72: template matching loses score at
                                       # different angles — TextNormalizer corrects rotation
                                       # after the crop, so a looser threshold is safe here.
POSITION_LOCK_SEARCH_MARGIN  = 80     # px around last position for fast search
POSITION_LOCK_BLUR_THRESHOLD = 30.0   # Laplacian variance below this = skip frame; 0 = disabled

# ---- Text rotation normalizer ---------------------------------------
# Corrects in-plane rotation of the text crop so that reference and live
# are always compared at the same orientation, regardless of the roller angle
# at the moment of capture.  Uses PCA on ink pixels (~3 ms on Pi CM5).
TEXT_NORM_ENABLED   = True
TEXT_NORM_MIN_ANGLE = 1.5   # degrees; skip correction below this (no perceptible effect)

# ---- Illumination correction ----------------------------------------
# Measures median brightness of the background (non-text) region in both
# reference and live, then applies an additive offset to the live frame
# before any diff is computed.  Eliminates false positives from lighting drift.
ILLUMINATION_CORRECT_ENABLED = True

# ---- Addition detection (PRIMARY defect gate) -----------------------
# Detects new dark marks added to the text region: drawn lines, smudges,
# debris resting on characters.  Uses a DIRECTIONAL diff so only pixels
# where live is darker than reference are counted — making the detector
# completely insensitive to faded/missing ink, lighting changes, or
# positional/rotational shifts after alignment.
# This is the ONLY signal that sets is_defect = True.
ADDITION_DETECTION_ENABLED = True
ADDITION_THRESHOLD         = 20   # intensity units in locally-normalised space.
                                   # After local-mean subtraction camera noise is ±3–6;
                                   # real marks produce 40–120 units.  Lower than before
                                   # because the normalisation removes lighting drift so
                                   # a tighter value catches faint marks without FP.
ADDITION_MIN_AREA          = 25   # px² — minimum blob area.  A 2×13 px line = 26 px².
ADDITION_BLUR_KSIZE        = 5    # Gaussian blur kernel before threshold (must be odd).
                                   # Averages out per-frame sensor noise spikes.
                                   # Set to 1 to disable blur.
LOCAL_NORM_KSIZE           = 41   # Gaussian kernel for local-mean subtraction applied
                                   # before the directional diff.  Removes slow illumination
                                   # gradients (roller curvature, lamp angle) while keeping
                                   # sharp local features intact.  Must be odd; increase for
                                   # larger ROIs or broader gradient patterns.

# ---- Background debris detection (OFF by default) -------------------
# Compact-blob search in the non-text region.  Not needed for the primary
# use case (debris on text); enable only if background contamination matters.
DEBRIS_DETECTION_ENABLED    = False
DEBRIS_MIN_AREA             = 20
DEBRIS_MAX_AREA             = 3000
DEBRIS_DIFF_THRESHOLD       = 22
DEBRIS_CIRCULARITY_MIN      = 0.20
DEBRIS_MAX_ASPECT_RATIO     = 3.5

# ---- Feature-based alignment (rotation-invariant) -------------------
# Replaces phase-correlation aligner for crops with rotational variance.
# Falls back to phase-correlation if ORB finds too few keypoints.
FEATURE_ALIGNER_ENABLED  = True
ORB_N_FEATURES           = 500    # Max ORB keypoints per image
MIN_MATCH_COUNT          = 10     # Min RANSAC inliers to accept homography
HOMOGRAPHY_RANSAC_THRESH = 5.0    # Reprojection error threshold (px)

# ---- Text mask inspection -------------------------------------------
# Focus pixel-diff and edge-diff scoring on ink pixels only.
# Scores are then normalised against text-pixel count, not total ROI area.
TEXT_MASK_ENABLED       = True
TEXT_MASK_DILATE_ITER   = 2       # Dilation iterations to capture edge anti-alias pixels
TEXT_MASK_MIN_CHAR_AREA = 20      # Min connected-component area (px²) = one character

# Per-character ink-density change threshold: fraction of ink change
# that flags a single character as defective (faded, missing, broken).
# 0.20 = 20 % ink-density change in a character region.
CHAR_INK_CHANGE_THRESHOLD = 0.20

# ---- Logging ------------------------------------------------
LOG_ENABLED = True
LOG_DIR = "logs"
SNAPSHOT_ON_DEFECT = False   # Auto-save ROI image on every confirmed defect
