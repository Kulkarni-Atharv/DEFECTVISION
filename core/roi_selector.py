from __future__ import annotations
import cv2
import numpy as np


# Maximum display width (pixels) — large frames are scaled down for the selector
_MAX_DISPLAY_W = 900


class ROISelector:
    """
    ROI selection via Tkinter, which talks to X11 directly and bypasses
    Qt (the source of the NULL-window-handler crash on Pi OS Bookworm).

    Drag a rectangle over the print region, then press ENTER or SPACE.
    Press Q or Escape to cancel.

    Requires: sudo apt install -y python3-tk
    Pillow (ImageTk) is used for image display — already a scikit-image dep.
    """

    def select(self, cam) -> tuple[int, int, int, int] | None:
        # Flush stale frames from the camera buffer
        for _ in range(5):
            cam.read()

        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Could not read frame for ROI selection.")
            return None

        return self._select_tk(frame)

    # ------------------------------------------------------------------

    def _select_tk(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        try:
            import tkinter as tk
        except ImportError:
            print("[ERROR] python3-tk not found.")
            print("        sudo apt install -y python3-tk")
            print("        Then rerun, or use:  python main.py --roi X Y W H")
            return None

        try:
            from PIL import Image, ImageTk
        except ImportError:
            print("[ERROR] Pillow ImageTk not available.")
            print("        sudo apt install -y python3-pil.imagetk")
            print("        Then rerun, or use:  python main.py --roi X Y W H")
            return None

        orig_h, orig_w = frame_bgr.shape[:2]
        scale  = min(1.0, _MAX_DISPLAY_W / orig_w)
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if scale < 1.0:
            frame_rgb = cv2.resize(frame_rgb, (disp_w, disp_h))

        # ---- Tkinter state ------------------------------------------
        roi_out  = [None]   # written by on_release
        start_xy = [None]
        rect_id  = [None]

        # ---- Build window -------------------------------------------
        root = tk.Tk()
        root.title("Select ROI  |  Drag → ENTER/SPACE confirm  |  Q/Esc cancel")
        root.resizable(False, False)

        canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="crosshair",
                           highlightthickness=0)
        canvas.pack()

        photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        status = tk.Label(root,
                          text="Drag with mouse to select the print region",
                          fg="#00cc44", bg="#1e1e1e",
                          font=("Helvetica", 11), pady=4)
        status.pack(fill=tk.X)

        # ---- Mouse callbacks ----------------------------------------
        def on_press(e):
            start_xy[0] = (e.x, e.y)
            if rect_id[0]:
                canvas.delete(rect_id[0])

        def on_drag(e):
            if not start_xy[0]:
                return
            if rect_id[0]:
                canvas.delete(rect_id[0])
            rect_id[0] = canvas.create_rectangle(
                start_xy[0][0], start_xy[0][1], e.x, e.y,
                outline="#00ff55", width=2,
            )

        def on_release(e):
            if not start_xy[0]:
                return
            x1 = min(start_xy[0][0], e.x)
            y1 = min(start_xy[0][1], e.y)
            x2 = max(start_xy[0][0], e.x)
            y2 = max(start_xy[0][1], e.y)
            if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                # Scale coordinates back to original image space
                roi_out[0] = (
                    int(x1 / scale),
                    int(y1 / scale),
                    int((x2 - x1) / scale),
                    int((y2 - y1) / scale),
                )
                status.config(
                    text=f"ROI {roi_out[0][2]}×{roi_out[0][3]} px  — press ENTER to confirm"
                )

        def confirm(e=None):
            if roi_out[0]:
                root.quit()

        def cancel(e=None):
            roi_out[0] = None
            root.quit()

        canvas.bind("<ButtonPress-1>",  on_press)
        canvas.bind("<B1-Motion>",       on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.bind("<Return>",            confirm)
        root.bind("<space>",             confirm)
        root.bind("q",                   cancel)
        root.bind("<Escape>",            cancel)

        root.mainloop()
        try:
            root.destroy()
        except tk.TclError:
            pass

        if roi_out[0]:
            x, y, w, h = roi_out[0]
            print(f"[INFO] ROI selected: x={x} y={y} w={w} h={h}")

        return roi_out[0]
