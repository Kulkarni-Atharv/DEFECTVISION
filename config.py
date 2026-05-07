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
# Max fraction of ROI size allowed as shift — rejects unreliable correlations
ALIGN_MAX_SHIFT_RATIO = 0.08  # 8 % of ROI width/height

# ---- Inspection thresholds ----------------------------------
# SSIM: 0 = completely different, 1 = identical
SSIM_THRESHOLD = 0.82
SSIM_WIN_SIZE = 7            # Window size for local SSIM; smaller catches finer defects

# Edge difference: fraction of pixels whose edges differ
EDGE_DIFF_THRESHOLD = 0.08

# Raw pixel difference: absolute intensity difference treated as defective
PIXEL_DIFF_THRESHOLD = 30    # 0–255

# ---- Defect scoring (weighted combination) ------------------
SSIM_WEIGHT = 0.60
EDGE_WEIGHT = 0.25
PIXEL_WEIGHT = 0.15

# Combined defect score: 0.0 = perfect, 1.0 = severe defect
DEFECT_SCORE_THRESHOLD = 0.30

# ---- Temporal consistency filter ----------------------------
# Prevents single noisy frames from triggering false alarms.
TEMPORAL_WINDOW = 8          # Number of frames in rolling window
TEMPORAL_DEFECT_RATIO = 0.60 # Fraction of window frames that must flag defect

# ---- Visualization ------------------------------------------
HEATMAP_ALPHA = 0.45         # 0 = no heatmap, 1 = full heatmap overlay
ROI_BORDER_THICKNESS = 3
PANEL_CELL_SCALE = 2.5       # Display scale multiplier for each panel cell
CORNER_ACCENT_LENGTH = 18    # Length of corner bracket lines on main feed

# ---- Logging ------------------------------------------------
LOG_ENABLED = True
LOG_DIR = "logs"
SNAPSHOT_ON_DEFECT = False   # Auto-save ROI image on every confirmed defect
