"""
DefectVision — Real-time print defect inspection
================================================
Entry point.  Run with:
    python main.py
    python main.py --roi 100 50 400 200   # skip GUI ROI selector

Key bindings (during inspection):
    Q      — quit
    R      — recapture reference (place clean sample first)
    S      — save snapshot of current live ROI to logs/
    SPACE  — pause / resume
"""
from __future__ import annotations
import argparse
import sys
import time
import cv2
import numpy as np

# --- project imports ------------------------------------------------
from config import (
    CAMERA_BACKEND,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    PICAMERA2_WIDTH, PICAMERA2_HEIGHT, PICAMERA2_FPS, PICAMERA2_WARMUP_S,
    REFERENCE_FRAME_COUNT, REFERENCE_WARMUP_FRAMES,
    LOG_DIR,
)
from core.camera          import create_camera
from core.roi_selector    import ROISelector
from core.preprocessor    import Preprocessor
from core.aligner         import Aligner
from core.inspector       import Inspector
from core.temporal_filter import TemporalFilter
from core.visualizer      import Visualizer
from utils.logger         import DefectLogger

import os
os.makedirs(LOG_DIR, exist_ok=True)

def _grab_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y: y + h, x: x + w].copy()


# ====================================================================
# Reference capture
# ====================================================================

def capture_reference(
    cap,
    roi: tuple[int, int, int, int],
    preprocessor: Preprocessor,
    n: int = REFERENCE_FRAME_COUNT,
) -> np.ndarray:
    """
    Average N preprocessed frames to obtain a stable, noise-reduced
    reference image.  Averaging in float64 then converting back to uint8
    preserves fine detail that matters for micro-defect detection.
    """
    accumulator: np.ndarray | None = None
    collected = 0

    while collected < n:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = preprocessor.process(_grab_roi(frame, roi))
        if accumulator is None:
            accumulator = np.float64(gray)
        else:
            accumulator += gray
        collected += 1
        time.sleep(1.0 / CAMERA_FPS)

    return np.uint8(accumulator / n)


# ====================================================================
# Reference capture UI
# ====================================================================

def wait_for_reference_capture(
    cap,
    roi: tuple[int, int, int, int],
    preprocessor: Preprocessor,
) -> np.ndarray | None:
    """
    Show a live preview with the ROI highlighted.
    The user places a clean sample and presses SPACE to capture.
    Returns the reference grayscale array or None on abort.
    """
    WIN = "DefectVision — Reference Capture  [SPACE=capture] [Q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    x, y, w, h = roi

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 255), 2)
        cv2.putText(display,
                    "Place CLEAN sample under camera  —  press SPACE to capture reference",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            print(f"[INFO] Capturing {REFERENCE_FRAME_COUNT} reference frames …")
            ref = capture_reference(cap, roi, preprocessor)
            cv2.destroyWindow(WIN)
            return ref
        elif key == ord('q'):
            cv2.destroyWindow(WIN)
            return None


# ====================================================================
# Main inspection loop
# ====================================================================

def run_inspection(
    cap,
    roi: tuple[int, int, int, int],
    ref_gray: np.ndarray,
    preprocessor: Preprocessor,
    aligner: Aligner,
    inspector: Inspector,
    temporal: TemporalFilter,
    visualizer: Visualizer,
    logger: DefectLogger,
) -> None:
    inspector.set_reference(ref_gray)
    temporal.reset()

    WIN_MAIN  = "DefectVision — Live Feed"
    WIN_PANEL = "DefectVision — Inspection Panel"
    cv2.namedWindow(WIN_MAIN,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_PANEL, cv2.WINDOW_NORMAL)

    frame_num   = 0
    fps         = 0.0
    fps_t0      = time.monotonic()
    fps_counter = 0

    # Dummy result for the first few frames while the temporal window fills
    from core.inspector import InspectionResult
    result           = InspectionResult()
    smoothed_score   = 0.0
    confirmed_defect = False
    paused           = False

    print("[INFO] Inspection running.  Q=quit  R=new reference  S=snapshot  SPACE=pause")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_num   += 1
            fps_counter += 1

            # FPS measurement
            if fps_counter >= 30:
                fps = fps_counter / max(time.monotonic() - fps_t0, 1e-6)
                fps_t0      = time.monotonic()
                fps_counter = 0

            # Extract and preprocess live ROI
            roi_bgr  = _grab_roi(frame, roi)
            live_gray = preprocessor.process(roi_bgr)

            # Align to compensate for sub-pixel vibration / position jitter
            live_aligned = aligner.align(ref_gray, live_gray)

            # Structural inspection
            result = inspector.inspect(ref_gray, live_aligned)

            # Temporal consistency — suppress single-frame noise
            warming_up = not temporal.window_full
            smoothed_score, confirmed_defect = temporal.update(
                result.defect_score, result.is_defect
            )
            if warming_up:
                confirmed_defect = False

            # Log
            logger.log(frame_num, result, confirmed_defect, smoothed_score, roi_bgr)

            # ---- Main feed display --------------------------------
            main_display = frame.copy()
            main_display = visualizer.draw_main_overlay(
                main_display, roi, confirmed_defect, smoothed_score, warming_up
            )
            cv2.putText(main_display,
                        f"FPS: {fps:.1f}  Frame: {frame_num}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            cv2.imshow(WIN_MAIN, main_display)

            # ---- Inspection panel ---------------------------------
            panel = visualizer.build_panel(
                roi_bgr, ref_gray, live_aligned,
                result, confirmed_defect, smoothed_score, fps, warming_up
            )
            cv2.imshow(WIN_PANEL, panel)

        else:
            # Paused — keep windows alive
            cv2.waitKey(50)

        # ---- Key handling ----------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")

        elif key == ord('r'):
            print("[INFO] Recapturing reference — place CLEAN sample, then press SPACE …")
            new_ref = wait_for_reference_capture(cap, roi, preprocessor)
            if new_ref is not None:
                ref_gray = new_ref
                inspector.set_reference(ref_gray)
                temporal.reset()
                print("[INFO] Reference updated.")
            else:
                print("[INFO] Reference recapture cancelled.")

        elif key == ord('s'):
            import cv2 as _cv2
            snap_path = os.path.join(
                LOG_DIR,
                f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            )
            _cv2.imwrite(snap_path, roi_bgr)
            print(f"[INFO] Snapshot saved: {snap_path}")

    cv2.destroyAllWindows()


# ====================================================================
# Entry point
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DefectVision print defect inspection")
    parser.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Skip GUI ROI selector and use fixed coordinates  e.g. --roi 100 50 400 200"
    )
    args = parser.parse_args()

    # ---- Camera ----------------------------------------------------
    print(f"[INFO] Starting camera (backend={CAMERA_BACKEND}) …")
    cam = create_camera(
        CAMERA_BACKEND,
        index    = CAMERA_INDEX,
        width    = PICAMERA2_WIDTH  if CAMERA_BACKEND.upper() == "PICAMERA2" else CAMERA_WIDTH,
        height   = PICAMERA2_HEIGHT if CAMERA_BACKEND.upper() == "PICAMERA2" else CAMERA_HEIGHT,
        fps      = PICAMERA2_FPS    if CAMERA_BACKEND.upper() == "PICAMERA2" else CAMERA_FPS,
        warmup_s = PICAMERA2_WARMUP_S,
        warmup_frames = REFERENCE_WARMUP_FRAMES,
    )
    if not cam.is_opened():
        print("[ERROR] Could not open camera.  Check CAMERA_BACKEND / CAMERA_INDEX in config.py.")
        sys.exit(1)

    w, h = cam.get_resolution()
    print(f"[INFO] Camera ready: {w}×{h} @ {cam.get_fps():.0f} fps")

    # ---- ROI selection ---------------------------------------------
    if args.roi:
        roi = tuple(args.roi)  # (x, y, w, h) from CLI
        print(f"[INFO] ROI from CLI: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
    else:
        print("[INFO] Select the print ROI on the live feed.")
        roi = ROISelector().select(cam)
        if roi is None:
            print("[INFO] ROI selection cancelled.  Exiting.")
            cam.release()
            sys.exit(0)

    x, y, w, h = roi
    print(f"[INFO] ROI: x={x} y={y} w={w} h={h}")

    # ---- Subsystem init --------------------------------------------
    preprocessor = Preprocessor()
    aligner      = Aligner()
    inspector    = Inspector()
    temporal     = TemporalFilter()
    visualizer   = Visualizer()
    logger       = DefectLogger()

    # ---- Reference capture -----------------------------------------
    ref_gray = wait_for_reference_capture(cam, roi, preprocessor)
    if ref_gray is None:
        print("[INFO] Reference capture cancelled.  Exiting.")
        cam.release()
        sys.exit(0)

    print(f"[INFO] Reference shape: {ref_gray.shape}  dtype: {ref_gray.dtype}")

    # ---- Inspection loop -------------------------------------------
    try:
        run_inspection(
            cam, roi, ref_gray,
            preprocessor, aligner, inspector, temporal, visualizer, logger,
        )
    finally:
        cam.release()
        summary = logger.summary()
        print("\n[SESSION SUMMARY]")
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
