import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from core import manual_capture_utils as mcu
from core.camera_utils import open_camera, read_frame
from core.infer import CPXDetector
import theme
import threading


class ZoomableCanvas(tk.Canvas):
    """A canvas that displays a frame with mouse-wheel zoom (centered on
    the cursor) and maps click coordinates back to the ORIGINAL frame's
    pixel space regardless of current zoom/pan. Zoom only affects what's
    displayed — it never affects the underlying saved image resolution.

    Also resizes its display area dynamically to fill whatever space
    Tkinter's geometry manager gives it (via <Configure>), instead of
    staying locked to a small fixed pixel size — so maximizing/resizing
    the window actually makes the camera preview bigger."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.display_w = 640
        self.display_h = 480
        self.zoom = 1.0
        self.center_x = None
        self.center_y = None
        self.frame_w = None
        self.frame_h = None
        self._last_shown = None  # keep last frame so resize can redraw immediately
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        if event.width > 10 and event.height > 10:
            self.display_w = event.width
            self.display_h = event.height
            if self._last_shown is not None:
                img, draw_fn = self._last_shown
                self.show(img, draw_fn=draw_fn, _store=False)

    def set_frame_size(self, w, h):
        if self.frame_w != w or self.frame_h != h:
            self.frame_w, self.frame_h = w, h
            self.center_x, self.center_y = w / 2, h / 2
            self.zoom = 1.0

    def reset_zoom(self):
        if self.frame_w:
            self.center_x, self.center_y = self.frame_w / 2, self.frame_h / 2
        self.zoom = 1.0

    def _on_wheel(self, event):
        if self.frame_w is None:
            return
        fx, fy = self.canvas_to_frame(event.x, event.y)
        factor = 1.2 if event.delta > 0 else (1 / 1.2)
        self.zoom = max(1.0, min(10.0, self.zoom * factor))
        self.center_x, self.center_y = fx, fy
        self.event_generate("<<ZoomChanged>>")

    def _crop_box(self):
        crop_w = self.frame_w / self.zoom
        crop_h = self.frame_h / self.zoom
        x0 = self.center_x - crop_w / 2
        y0 = self.center_y - crop_h / 2
        x0 = max(0, min(self.frame_w - crop_w, x0))
        y0 = max(0, min(self.frame_h - crop_h, y0))
        return x0, y0, x0 + crop_w, y0 + crop_h

    def canvas_to_frame(self, cx, cy):
        if self.frame_w is None:
            return cx, cy
        x0, y0, x1, y1 = self._crop_box()
        fx = x0 + (cx / self.display_w) * (x1 - x0)
        fy = y0 + (cy / self.display_h) * (y1 - y0)
        return fx, fy

    def show(self, bgr_img, draw_fn=None, _store=True):
        """draw_fn(img) -> img: optional callback to draw overlays onto
        the FULL-RESOLUTION frame before cropping/resizing for display,
        so overlays stay pixel-accurate at any zoom/window size."""
        self.set_frame_size(bgr_img.shape[1], bgr_img.shape[0])

        if _store:
            self._last_shown = (bgr_img, draw_fn)

        display = bgr_img
        if draw_fn is not None:
            display = draw_fn(bgr_img.copy())

        x0, y0, x1, y1 = self._crop_box()
        crop = display[int(y0):int(y1), int(x0):int(x1)]
        if crop.size == 0:
            crop = display

        # Preserve aspect ratio inside whatever space we've been given,
        # instead of stretching the image to fill a mismatched box.
        crop_h, crop_w = crop.shape[:2]
        scale = min(self.display_w / crop_w, self.display_h / crop_h)
        target_w = max(1, int(crop_w * scale))
        target_h = max(1, int(crop_h * scale))

        resized = cv2.resize(crop, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.imgtk = imgtk
        self.delete("all")
        # Center the image within the canvas if there's leftover space
        # (e.g. window is wider than the image's aspect ratio allows).
        off_x = (self.display_w - target_w) // 2
        off_y = (self.display_h - target_h) // 2
        self.create_image(off_x, off_y, anchor="nw", image=imgtk)
        self._img_offset = (off_x, off_y, scale, x0, y0)

    def canvas_to_frame(self, cx, cy):
        if self.frame_w is None:
            return cx, cy
        if getattr(self, "_img_offset", None) is None:
            x0, y0, x1, y1 = self._crop_box()
            fx = x0 + (cx / self.display_w) * (x1 - x0)
            fy = y0 + (cy / self.display_h) * (y1 - y0)
            return fx, fy
        off_x, off_y, scale, x0, y0 = self._img_offset
        fx = x0 + (cx - off_x) / scale
        fy = y0 + (cy - off_y) / scale
        return fx, fy



class ManualRecordWindow(tk.Toplevel):
    """Phase 1: plain continuous recording — every displayed frame gets
    saved raw (no labeling, no pausing) while the user physically moves
    the product through whatever angles/poses they want covered. Mouse
    wheel zooms the PREVIEW only; saved frames are always full camera
    resolution regardless of zoom."""

    def __init__(self, master, product_name):
        super().__init__(master)
        self.title(f"Record — {product_name}")
        theme.style_window(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        # Start large — near-fullscreen — so the camera preview is big
        # and readable, instead of a small fixed box in a mostly-empty
        # window.
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{int(sw*0.9)}x{int(sh*0.9)}+{int(sw*0.05)}+{int(sh*0.05)}")
        self.minsize(700, 620)

        self.product_name = product_name
        self._closed = False
        self.cap = open_camera()
        self.recording = False
        self.batch_path = None
        self.frame_count = 0

        self.canvas = ZoomableCanvas(self, bg=theme.BG_PANEL)
        self.canvas.pack(padx=10, pady=10, fill="both", expand=True)


        tk.Label(self, text="Scroll to zoom the preview (doesn't affect saved resolution).",
                 bg=theme.BG_DARK, fg=theme.CREAM, font=("Consolas", 9)).pack()

        self.status_label = tk.Label(self, text="Not recording.", bg=theme.BG_DARK,
                                      fg=theme.ACCENT_YELLOW, font=theme.FONT_LABEL)
        self.status_label.pack(pady=(5, 5))

        btn_row = tk.Frame(self, bg=theme.BG_DARK)
        btn_row.pack(pady=10)

        self.record_btn = tk.Button(btn_row, text="● Start recording", command=self.toggle_recording)
        theme.style_button(self.record_btn)
        self.record_btn.pack(side="left", padx=5)

        self.finish_btn = tk.Button(btn_row, text="Auto-label & Review this batch",
                                     command=self.run_autolabel_and_review, state="disabled")
        theme.style_button(self.finish_btn)
        self.finish_btn.pack(side="left", padx=5)

        close_btn = tk.Button(btn_row, text="Close", command=self.on_close)
        theme.style_button(close_btn)
        close_btn.pack(side="left", padx=5)

        self.update_frame()

    def toggle_recording(self):
        if not self.recording:
            self.batch_path = mcu.new_temp_batch(self.product_name)
            self.frame_count = 0
            self.recording = True
            self.record_btn.configure(text="■ Stop recording", bg=theme.ACCENT_YELLOW)
            self.finish_btn.configure(state="disabled")
            self.status_label.configure(text="Recording... move the product around freely.")
        else:
            self.recording = False
            self.record_btn.configure(text="● Start recording", bg=theme.ACCENT_ORANGE)
            self.finish_btn.configure(state="normal" if self.frame_count > 0 else "disabled")
            self.status_label.configure(
                text=f"Stopped. Captured {self.frame_count} raw frame(s). "
                     f"Click 'Auto-label & Review' when ready.")

    def update_frame(self):
        if self._closed:
            return
        ret, frame = read_frame(self.cap)

        if ret:
            if self.recording:
                fname = f"{self.frame_count:05d}.jpg"
                cv2.imwrite(os.path.join(self.batch_path, "images", fname), frame)
                self.frame_count += 1
                self.status_label.configure(
                    text=f"Recording... {self.frame_count} frame(s) captured.")

            self.canvas.show(frame)

        self.after(30, self.update_frame)

    def run_autolabel_and_review(self):
        if self.batch_path is None or self.frame_count == 0:
            return
        if self.recording:
            self.toggle_recording()

        detector = CPXDetector(conf_threshold=0.5)
        use_model = detector.loaded
        if not detector.loaded:
            proceed = messagebox.askyesno(
                "No model loaded",
                f"No trained model is available: {detector.load_error}\n\n"
                f"You can still proceed and label every frame manually with SAM/points "
                f"in the review step. Continue?"
            )
            if not proceed:
                return
        elif not detector.knows_label(self.product_name):
            # The current model was never trained on this product: its
            # "best detection" would be some OTHER object, so skip
            # auto-labeling and go straight to click-to-label review.
            use_model = messagebox.askyesno(
                "Model doesn't know this product",
                f"The current model was not trained on '{self.product_name}'.\n\n"
                f"Yes = try auto-labeling anyway\n"
                f"No = skip it and label every frame by clicking (SAM)"
            )

        self.finish_btn.configure(state="disabled")
        self.record_btn.configure(state="disabled")

        win = tk.Toplevel(self)
        win.title("Auto-labeling batch...")
        theme.style_window(win)
        win.geometry("400x140")
        win.resizable(False, False)

        status = tk.Label(win, text="Starting...", bg=theme.BG_DARK, fg=theme.CREAM,
                           font=theme.FONT_LABEL)
        status.pack(pady=15)
        progress = ttk.Progressbar(win, orient="horizontal", length=360, mode="determinate")
        progress.pack(pady=5)

        def on_progress(done, total):
            def update_ui():
                progress["value"] = (done / total) * 100
                status.configure(text=f"Labeling {done}/{total}...")
            win.after(0, update_ui)

        def finish_ui(labeled, total):
            win.destroy()
            messagebox.showinfo(
                "Auto-labeling complete",
                f"{labeled}/{total} frame(s) got an automatic mask.\n"
                f"{total - labeled} frame(s) need manual fixing in the review step."
            )
            ManualReviewWindow(self.master, self.batch_path, self.product_name)
            self.on_close()

        if not use_model:
            self.after(100, lambda: finish_ui(0, self.frame_count))
            return

        # Run the model in a worker thread so the progress bar actually
        # updates (previously this ran on the UI thread and froze it).
        def worker():
            labeled, total = mcu.auto_label_batch(
                self.batch_path, detector, conf_threshold=0.5, progress_callback=on_progress)
            self.after(100, lambda: finish_ui(labeled, total))

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self):
        self._closed = True
        self.recording = False
        self.cap.release()
        self.destroy()


class ManualReviewWindow(tk.Toplevel):
    """Phase 2: scrub through captured frames using arrow keys (no nav
    buttons), see the auto-labeled mask or 'NO MASK' if the model missed
    it. Left-click = foreground point, right-click = background point —
    EVERY click immediately re-runs SAM and immediately overwrites that
    frame's label on disk. No separate 'preview'/'accept' step. Moving
    to the next frame (arrow key) is the entire confirmation step.
    Nothing reaches the real dataset until 'Commit batch' is pressed."""

    def __init__(self, master, batch_path, product_name):
        super().__init__(master)
        self.title(f"Review batch — {product_name}")
        theme.style_window(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{int(sw*0.9)}x{int(sh*0.9)}+{int(sw*0.05)}+{int(sh*0.05)}")
        self.minsize(820, 720)


        self.batch_path = batch_path
        self.product_name = product_name
        self.sam_helper = None

        self.frames = mcu.list_batch_frames(batch_path)
        self.index = 0
        self.click_points = []          # accumulated points for CURRENT frame
        self.original_polygon = None    # what auto-labeling produced, for "Clear points"
        self.current_frame_img = None

        if not self.frames:
            tk.Label(self, text="No frames in this batch.", bg=theme.BG_DARK,
                     fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=40)
            close_btn = tk.Button(self, text="Close", command=self.destroy)
            theme.style_button(close_btn)
            close_btn.pack(pady=10)
            return

        self.canvas = ZoomableCanvas(self, bg=theme.BG_PANEL)
        self.canvas.pack(padx=10, pady=10, fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<<ZoomChanged>>", lambda e: self.redraw())

        tk.Label(self, text="Left-click = foreground point, right-click = background point. "
                             "Each click immediately updates & saves the mask.\n"
                             "Scroll to zoom for precise clicking. "
                             "Arrow keys: ← → step 1 frame, Shift+← → step by Skip amount.",
                 bg=theme.BG_DARK, fg=theme.ACCENT_YELLOW, font=("Consolas", 9),
                 justify="center").pack(pady=(0, 5))

        self.info_label = tk.Label(self, text="", bg=theme.BG_DARK, fg=theme.CREAM,
                                    font=theme.FONT_LABEL)
        self.info_label.pack(pady=(0, 5))

        nav_row = tk.Frame(self, bg=theme.BG_DARK)
        nav_row.pack(pady=3)
        tk.Label(nav_row, text="Skip amount:", bg=theme.BG_DARK, fg=theme.CREAM,
                 font=theme.FONT_LABEL).pack(side="left", padx=(0, 5))
        self.skip_var = tk.IntVar(value=10)
        tk.Entry(nav_row, textvariable=self.skip_var, width=6,
                 bg=theme.BG_PANEL, fg=theme.CREAM,
                 insertbackground=theme.CREAM, font=theme.FONT_LABEL).pack(side="left")

        reset_zoom_btn = tk.Button(nav_row, text="Reset zoom", command=self.reset_zoom)
        theme.style_button(reset_zoom_btn)
        reset_zoom_btn.pack(side="left", padx=15)

        clear_pts_btn = tk.Button(nav_row, text="Clear points (revert)", command=self.clear_points)
        theme.style_button(clear_pts_btn)
        clear_pts_btn.pack(side="left", padx=5)

        remove_btn = tk.Button(nav_row, text="🗑 Remove frame", command=self.remove_current_frame)
        theme.style_button(remove_btn)
        remove_btn.pack(side="left", padx=5)

        self.scrub_var = tk.IntVar(value=0)
        self.scrub_slider = tk.Scale(
            self, from_=0, to=len(self.frames) - 1, orient="horizontal",
            variable=self.scrub_var, bg=theme.BG_DARK, fg=theme.CREAM,
            troughcolor=theme.BG_PANEL, highlightthickness=0, length=780,
            showvalue=False, command=self.on_scrub
        )
        self.scrub_slider.pack(pady=5)

        # --- Commit / discard, bottom of window ---
        tk.Frame(self, bg=theme.CREAM, height=1).pack(fill="x", padx=10, pady=10)

        finalize_row = tk.Frame(self, bg=theme.BG_DARK)
        finalize_row.pack(pady=8)

        commit_btn = tk.Button(finalize_row, text="✓ Commit batch to training data",
                                command=self.commit_batch)
        theme.style_button(commit_btn)
        commit_btn.pack(side="left", padx=5)

        discard_btn = tk.Button(finalize_row, text="✗ Discard entire batch",
                                 command=self.discard_batch)
        theme.style_button(discard_btn)
        discard_btn.pack(side="left", padx=5)

        # Arrow key navigation
        self.bind("<Left>", lambda e: self.jump(-1))
        self.bind("<Right>", lambda e: self.jump(1))
        self.bind("<Shift-Left>", lambda e: self.jump(-self.skip_var.get()))
        self.bind("<Shift-Right>", lambda e: self.jump(self.skip_var.get()))
        self.focus_set()

        self.show_frame(0)

    # ------------------------------------------------------------------
    # Label read/write helpers
    # ------------------------------------------------------------------

    def _load_polygon(self, label_path, w_img, h_img):
        if not os.path.exists(label_path):
            return None
        try:
            with open(label_path, "r") as f:
                line = f.readline().strip()
            parts = line.split()
            if len(parts) < 7 or len(parts) % 2 == 0:
                return None
            coords = list(map(float, parts[1:]))
            pts = []
            for i in range(0, len(coords), 2):
                pts.append([coords[i] * w_img, coords[i + 1] * h_img])
            return np.array(pts, dtype=np.float32)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def show_frame(self, index):
        index = max(0, min(len(self.frames) - 1, index))
        self.index = index
        self.click_points = []

        info = self.frames[index]
        img = cv2.imread(info["image_path"])
        if img is None:
            self.info_label.configure(text=f"[Missing file: {info['image_path']}]")
            return

        self.current_frame_img = img
        h_img, w_img = img.shape[:2]
        self.original_polygon = self._load_polygon(info["label_path"], w_img, h_img)

        self.redraw()


        self.scrub_var.set(index)
        self.info_label.configure(
            text=f"[{index + 1}/{len(self.frames)}]  {os.path.basename(info['image_path'])}")






    def _current_polygon(self):
        """Returns whatever polygon is currently in the label file for
        this frame (reflects the latest SAM click result if any)."""
        info = self.frames[self.index]
        h_img, w_img = self.current_frame_img.shape[:2]
        return self._load_polygon(info["label_path"], w_img, h_img)

    def redraw(self):
        polygon = self._current_polygon()
        click_points = self.click_points

        def draw_fn(img):
            if polygon is not None and len(polygon) >= 3:
                pts = np.array(polygon, dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (0, 200, 0))
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            else:
                cv2.putText(img, "NO MASK - click to add points", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            for fx, fy, lbl in click_points:
                color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                cv2.circle(img, (int(fx), int(fy)), 6, color, -1)

            return img

        self.canvas.show(self.current_frame_img, draw_fn=draw_fn)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def jump(self, delta):
        self.show_frame(self.index + delta)

    def on_scrub(self, value):
        self.show_frame(int(value))

    def reset_zoom(self):
        self.canvas.reset_zoom()
        self.redraw()

    # ------------------------------------------------------------------
    # Click-to-refine (immediate SAM + immediate save)
    # ------------------------------------------------------------------

    def _get_sam(self):
        if self.sam_helper is None:
            from core.sam_utils import SAMHelper
            self.sam_helper = SAMHelper()
            if not self.sam_helper.loaded:
                messagebox.showwarning("SAM unavailable", self.sam_helper.error)
                return None
        return self.sam_helper



    def _add_point_and_refine(self, fx, fy, label):
        if getattr(self, "_sam_busy", False):
            # A previous click on this frame is still processing —
            # ignore this click rather than stacking up calls.
            return

        # Clicks in the letterbox area around the image map outside the
        # frame; clamp them so SAM always gets a valid pixel coordinate.
        h_img, w_img = self.current_frame_img.shape[:2]
        fx = min(max(0.0, fx), w_img - 1.0)
        fy = min(max(0.0, fy), h_img - 1.0)

        self.click_points.append((fx, fy, label))
        self.redraw()  # show the raw click point immediately, mask updates when SAM finishes

        sam = self._get_sam()
        if sam is None:
            return

        self._sam_busy = True
        self.config(cursor="watch")

        pts = [(x, y) for x, y, _ in self.click_points]
        labels = [lbl for _, _, lbl in self.click_points]
        frame_index_at_click = self.index
        frame_img = self.current_frame_img
        info = self.frames[self.index]

        def worker():
            polygon, error = sam.predict_from_points(frame_img, pts, labels)

            def apply_result():
                self._sam_busy = False
                self.config(cursor="")

                if error:
                    messagebox.showwarning(
                        "SAM issue",
                        f"{error}\n\nYour progress on all other frames is completely "
                        f"safe — this only affects the current click."
                    )
                    return

                if self.index == frame_index_at_click and polygon is not None and len(polygon) >= 3:
                    mcu.write_manual_label(info, polygon, frame_img.shape)
                    self.redraw()

            self.after(0, apply_result)

        threading.Thread(target=worker, daemon=True).start()


    

    def on_left_click(self, event):
        fx, fy = self.canvas.canvas_to_frame(event.x, event.y)
        self._add_point_and_refine(fx, fy, 1)
        self.focus_set()

    def on_right_click(self, event):
        fx, fy = self.canvas.canvas_to_frame(event.x, event.y)
        self._add_point_and_refine(fx, fy, 0)
        self.focus_set()


    def clear_points(self):
        """Reverts this frame back to whatever the auto-label originally
        produced (or no label, if the model never found anything here),
        discarding any manual SAM clicks made on this frame."""
        self.click_points = []
        info = self.frames[self.index]

        if self.original_polygon is not None and len(self.original_polygon) >= 3:
            mcu.write_manual_label(info, self.original_polygon, self.current_frame_img.shape)
        else:
            if os.path.exists(info["label_path"]):
                os.remove(info["label_path"])

        self.redraw()

    # ------------------------------------------------------------------
    # Frame removal
    # ------------------------------------------------------------------

    def remove_current_frame(self):
        info = self.frames[self.index]
        mcu.remove_frame(info)
        del self.frames[self.index]

        if not self.frames:
            messagebox.showinfo("Empty", "No frames left in this batch.")
            self.on_close()
            return

        self.scrub_slider.configure(to=len(self.frames) - 1)
        self.show_frame(min(self.index, len(self.frames) - 1))

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def commit_batch(self):
        labeled_count = sum(1 for f in self.frames if os.path.exists(f["label_path"]))
        unlabeled_count = len(self.frames) - labeled_count

        msg = f"Commit {labeled_count} labeled frame(s) to the real training dataset "
        msg += f"as a new session for '{self.product_name}'.\n\n"
        if unlabeled_count:
            msg += (f"{unlabeled_count} frame(s) still have NO mask and will be "
                    f"DROPPED (not committed) — fix or remove them first if you "
                    f"want to keep them.\n\n")
        msg += "Continue?"

        confirm = messagebox.askyesno("Confirm commit", msg)
        if not confirm:
            return

        session_path, count = mcu.commit_batch(self.batch_path, self.product_name)
        messagebox.showinfo("Committed", f"Committed {count} frame(s) to:\n{session_path}")
        self.on_close()

    def discard_batch(self):
        confirm = messagebox.askyesno(
            "Discard batch",
            "This will permanently delete every frame captured in this batch. "
            "Nothing will be added to your training dataset.\n\nContinue?"
        )
        if not confirm:
            return
        mcu.discard_batch(self.batch_path)
        self.on_close()

    def on_close(self):
        # Batch stays on disk in the temp folder if closed without commit/discard —
        # nothing is lost, it just isn't finalized yet.
        if self.sam_helper is not None:
            self.sam_helper.shutdown()
            self.sam_helper = None
        self.destroy()

