import cv2
import os
import subprocess
import platform
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
from core import dataset_utils as du
from core.camera_utils import open_camera
from core.record_settings import get_settings, save_settings
from core.infer import CPXDetector
import theme


class Tooltip:
    """Simple hover tooltip: shows a small popup with explanatory text
    when the mouse hovers over a widget, hides it on leave."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 15
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        self.tip.attributes("-topmost", True)
        label = tk.Label(self.tip, text=self.text, justify="left",
                          bg="#ffffe0", fg="black", relief="solid", borderwidth=1,
                          font=("Consolas", 9), wraplength=300)
        label.pack()

    def hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class CollapsibleSection(tk.Frame):
    """A titled frame with a small triangle toggle button that shows/hides
    its inner content frame."""
    def __init__(self, parent, title, start_expanded=True):
        super().__init__(parent, bg=theme.BG_PANEL)
        self.expanded = tk.BooleanVar(value=start_expanded)

        header = tk.Frame(self, bg=theme.BG_PANEL)
        header.pack(fill="x")

        self.toggle_btn = tk.Button(
            header, text=self._arrow(), width=2, relief="flat",
            bg=theme.BG_PANEL, fg=theme.ACCENT_ORANGE,
            activebackground=theme.BG_PANEL, activeforeground=theme.ACCENT_YELLOW,
            font=theme.FONT_LABEL, command=self.toggle, cursor="hand2", bd=0
        )
        self.toggle_btn.pack(side="left", padx=(4, 2), pady=4)

        tk.Label(header, text=title, bg=theme.BG_PANEL, fg=theme.CREAM,
                 font=theme.FONT_LABEL).pack(side="left", pady=4)

        self.content = tk.Frame(self, bg=theme.BG_PANEL)
        if start_expanded:
            self.content.pack(fill="x", padx=10, pady=(0, 8))

    def _arrow(self):
        return "▼" if self.expanded.get() else "▶"

    def toggle(self):
        self.expanded.set(not self.expanded.get())
        if self.expanded.get():
            self.content.pack(fill="x", padx=10, pady=(0, 8))
        else:
            self.content.pack_forget()
        self.toggle_btn.configure(text=self._arrow())


class ColorPickerWindow(tk.Toplevel):
    """Freezes the current camera frame and lets the user click on it to
    sample colors, for manual-mode color filtering."""
    def __init__(self, master, frame_bgr, on_sample, mode_label="exclude"):
        super().__init__(master)
        self.title(f"Click colors to {mode_label} (e.g. background/cardboard)")
        theme.style_window(self)
        self.on_sample = on_sample

        self.frame_bgr = frame_bgr
        self.display_w, self.display_h = 640, 480
        disp = cv2.resize(frame_bgr, (self.display_w, self.display_h))
        self.scale_x = frame_bgr.shape[1] / self.display_w
        self.scale_y = frame_bgr.shape[0] / self.display_h

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas = tk.Canvas(self, width=self.display_w, height=self.display_h,
                                 cursor="crosshair", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.create_image(0, 0, anchor="nw", image=self.imgtk)
        self.canvas.bind("<Button-1>", self.on_click)

        self.canvas.bind("<ButtonPress-3>", self.on_drag_start)
        self.canvas.bind("<B3-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-3>", self.on_drag_release)
        self._drag_start = None
        self._drag_rect_id = None

        tk.Label(self, text=f"Left-click: sample one pixel to {mode_label}.\n"
                             f"Right-click + drag: bulk-sample a region to {mode_label}.\n"
                             "Close this window when done.",
                 bg=theme.BG_DARK, fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=(0, 5))

        self.sample_list_label = tk.Label(self, text="Samples this session: 0",
                                           bg=theme.BG_DARK, fg=theme.ACCENT_YELLOW,
                                           font=theme.FONT_LABEL)
        self.sample_list_label.pack(pady=(0, 10))
        self._count = 0

        close_btn = tk.Button(self, text="Done", command=self.destroy)
        theme.style_button(close_btn)
        close_btn.pack(pady=(0, 10))

    def _add_sample(self, h, s, v):
        self.on_sample(h, s, v)
        self._count += 1
        self.sample_list_label.configure(text=f"Samples this session: {self._count}")

    def on_click(self, event):
        fx = int(event.x * self.scale_x)
        fy = int(event.y * self.scale_y)
        fx = max(0, min(self.frame_bgr.shape[1] - 1, fx))
        fy = max(0, min(self.frame_bgr.shape[0] - 1, fy))

        bgr_pixel = self.frame_bgr[fy, fx].reshape(1, 1, 3).astype(np.uint8)
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]
        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

        self._add_sample(h, s, v)
        self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                                 outline="lime", width=2)

    def on_drag_start(self, event):
        self._drag_start = (event.x, event.y)
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
        self._drag_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="cyan", width=2)

    def on_drag_motion(self, event):
        if self._drag_start is None or self._drag_rect_id is None:
            return
        x0, y0 = self._drag_start
        self.canvas.coords(self._drag_rect_id, x0, y0, event.x, event.y)

    def on_drag_release(self, event):
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None

        dx0, dx1 = sorted((max(0, min(x0, self.display_w)), max(0, min(x1, self.display_w))))
        dy0, dy1 = sorted((max(0, min(y0, self.display_h)), max(0, min(y1, self.display_h))))

        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

        if (dx1 - dx0) < 4 or (dy1 - dy0) < 4:
            return

        fx0 = int(dx0 * self.scale_x)
        fy0 = int(dy0 * self.scale_y)
        fx1 = int(dx1 * self.scale_x)
        fy1 = int(dy1 * self.scale_y)
        fx0 = max(0, min(self.frame_bgr.shape[1] - 1, fx0))
        fx1 = max(0, min(self.frame_bgr.shape[1], fx1))
        fy0 = max(0, min(self.frame_bgr.shape[0] - 1, fy0))
        fy1 = max(0, min(self.frame_bgr.shape[0], fy1))

        region_bgr = self.frame_bgr[fy0:fy1, fx0:fx1]
        if region_bgr.size == 0:
            return

        region_hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        bucket = 8
        quantized = (region_hsv.astype(np.int32) // bucket) * bucket
        flat = quantized.reshape(-1, 3)
        unique_colors = np.unique(flat, axis=0)

        max_new_samples = 40
        if len(unique_colors) > max_new_samples:
            step = len(unique_colors) // max_new_samples
            unique_colors = unique_colors[::step][:max_new_samples]

        for h, s, v in unique_colors:
            self._add_sample(int(h), int(s), int(v))

        self.canvas.create_rectangle(dx0, dy0, dx1, dy1, outline="lime", width=2)


class RecordWindow(tk.Toplevel):
    def __init__(self, master, product_name):
        super().__init__(master)
        self.title(f"Record training data — {product_name}")
        theme.style_window(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.geometry("940x900")
        self.minsize(940, 900)
        self.product_name = product_name
        self.recording = False
        self.session_path = None
        self.frame_idx = 0

        self.cap = open_camera()
        self.ai_detector = CPXDetector(conf_threshold=0.5)

        self.default_snapshot = get_settings(product_name)
        saved = self.default_snapshot

        # ------------------------------------------------------------------
        # Video preview area (fixed size, pinned at top)
        # ------------------------------------------------------------------
        video_frame = tk.Frame(self, bg=theme.BG_DARK)
        video_frame.pack(padx=10, pady=10, fill="x")

        PREVIEW_W, PREVIEW_H = 400, 300

        self.live_label = tk.Label(video_frame, bg=theme.BG_PANEL,
                                    width=PREVIEW_W, height=PREVIEW_H)
        self.live_label.grid(row=0, column=0, padx=5)
        self.live_label.grid_propagate(False)

        self.mask_label = tk.Label(video_frame, bg=theme.BG_PANEL,
                                    width=PREVIEW_W, height=PREVIEW_H)
        self.mask_label.grid(row=0, column=1, padx=5)
        self.mask_label.grid_propagate(False)

        tk.Label(video_frame, text="Live feed (saved, full frame)", bg=theme.BG_DARK,
                  fg=theme.CREAM, font=theme.FONT_LABEL).grid(row=1, column=0)
        tk.Label(video_frame, text="Mask preview", bg=theme.BG_DARK,
                  fg=theme.CREAM, font=theme.FONT_LABEL).grid(row=1, column=1)

        # ------------------------------------------------------------------
        # AI / Manual mode toggle (top right, with confirmation)
        # ------------------------------------------------------------------
        mode_bar = tk.Frame(self, bg=theme.BG_DARK)
        mode_bar.pack(fill="x", padx=10)

        self.use_ai_mode = tk.BooleanVar(value=self.ai_detector.loaded)

        self.ai_chk = tk.Checkbutton(
            mode_bar,
            text="AI-assisted mode (recommended)" if self.ai_detector.loaded else
                 "AI mode unavailable — no model loaded",
            variable=self.use_ai_mode, bg=theme.BG_DARK, fg=theme.CREAM,
            selectcolor=theme.BG_PANEL, font=theme.FONT_LABEL,
            state="normal" if self.ai_detector.loaded else "disabled",
            command=self._on_mode_toggle
        )
        self.ai_chk.pack(side="right", padx=5, pady=5)
        Tooltip(self.ai_chk, "When on, the trained model finds the product and its exact "
                              "mask outline directly — no sliders needed. Turn off only if "
                              "the model is struggling on a tricky product/background and "
                              "you need manual threshold/contour tuning instead.")

        # --- Color filtering controls (manual mode only) ---
        self.exclude_colors = list(saved.get("exclude_colors", []))
        self.include_colors = list(saved.get("include_colors", []))
        self.color_mode_var = tk.StringVar(value=saved.get("color_mode", "exclude"))

        self.manual_frame = tk.Frame(self, bg=theme.BG_DARK)

        color_mode_frame = tk.Frame(self.manual_frame, bg=theme.BG_DARK)
        color_mode_frame.pack(pady=(5, 0))

        excl_radio = tk.Radiobutton(
            color_mode_frame, text="Remove selected colors", variable=self.color_mode_var,
            value="exclude", bg=theme.BG_DARK, fg=theme.CREAM, selectcolor=theme.BG_PANEL,
            font=theme.FONT_LABEL, command=self._on_color_mode_change
        )
        excl_radio.pack(side="left", padx=5)

        incl_radio = tk.Radiobutton(
            color_mode_frame, text="Keep only selected colors", variable=self.color_mode_var,
            value="include", bg=theme.BG_DARK, fg=theme.CREAM, selectcolor=theme.BG_PANEL,
            font=theme.FONT_LABEL, command=self._on_color_mode_change
        )
        incl_radio.pack(side="left", padx=5)

        color_btn_frame = tk.Frame(self.manual_frame, bg=theme.BG_DARK)
        color_btn_frame.pack(pady=5)

        pick_color_btn = tk.Button(color_btn_frame, text="🎨 Pick colors",
                                    command=self.open_color_picker)
        theme.style_button(pick_color_btn)
        pick_color_btn.pack(side="left", padx=5)

        manage_btn = tk.Button(color_btn_frame, text="Manage colors",
                                command=self.open_color_manager)
        theme.style_button(manage_btn)
        manage_btn.pack(side="left", padx=5)

        clear_excl_btn = tk.Button(color_btn_frame, text="Clear all",
                                    command=self.clear_colors)
        theme.style_button(clear_excl_btn)
        clear_excl_btn.pack(side="left", padx=5)

        self.excl_count_label = tk.Label(
            color_btn_frame, text="", bg=theme.BG_DARK, fg=theme.CREAM, font=theme.FONT_LABEL)
        self.excl_count_label.pack(side="left", padx=10)
        self._update_color_count_label()

        # ------------------------------------------------------------------
        # Record / Reset / Close buttons — always visible
        # ------------------------------------------------------------------
        btn_frame = tk.Frame(self, bg=theme.BG_DARK)
        btn_frame.pack(pady=10)

        self.record_btn = tk.Button(btn_frame, text="● Record training data",
                                     command=self.toggle_record)
        theme.style_button(self.record_btn)
        self.record_btn.pack(side="left", padx=10)

        reset_btn = tk.Button(btn_frame, text="Reset to Default",
                               command=self.reset_to_default)
        theme.style_button(reset_btn)
        reset_btn.pack(side="left", padx=10)

        close_btn = tk.Button(btn_frame, text="Close", command=self.on_close)
        theme.style_button(close_btn)
        close_btn.pack(side="left", padx=10)

        self.status_label = tk.Label(self, text="Not recording", bg=theme.BG_DARK,
                                      fg=theme.ACCENT_YELLOW, font=theme.FONT_LABEL)
        self.status_label.pack(pady=(0, 5))

        # ------------------------------------------------------------------
        # AI confidence + margin controls (always visible, lightweight)
        # ------------------------------------------------------------------
        ai_settings_frame = tk.Frame(self, bg=theme.BG_PANEL)
        ai_settings_frame.pack(fill="x", padx=10, pady=(0, 6))

        self.ai_conf_var = tk.IntVar(value=int(saved.get("ai_conf", 50)))
        self.margin_var = tk.IntVar(value=saved.get("margin", 0))

        self._make_slider(ai_settings_frame, "AI conf threshold (%)", self.ai_conf_var, 1, 99, 0,
                           tip="Minimum model confidence required to accept a detection.")
        self._make_slider(ai_settings_frame, "Mask margin (px)", self.margin_var, -20, 40, 1,
                           tip="Pixels grown (or shrunk, if negative) around the AI mask "
                               "outline before saving. Usually 0 is correct now that the "
                               "model outputs a precise mask directly.")

        # ------------------------------------------------------------------
        # Manual mode settings — collapsed by default, only shown when
        # manual mode is active (hidden entirely in AI mode).
        # ------------------------------------------------------------------
        outer = tk.Frame(self, bg=theme.BG_DARK)
        self.manual_settings_outer = outer

        canvas = tk.Canvas(outer, bg=theme.BG_DARK, highlightthickness=0, height=260)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        sections_frame = tk.Frame(canvas, bg=theme.BG_DARK)
        canvas_window = canvas.create_window((0, 0), window=sections_frame, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        sections_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def on_mousewheel(event):
            if canvas.winfo_exists():
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        self.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self.thresh_var = tk.IntVar(value=saved["threshold"])
        self.blur_var = tk.IntVar(value=saved["blur"])
        self.blur_sigma_var = tk.IntVar(value=int(saved.get("blur_sigma", 0)))
        self.invert_var = tk.BooleanVar(value=saved["invert"])
        self.morph_var = tk.IntVar(value=saved["morph"])
        self.morph_op_var = tk.StringVar(value=saved.get("morph_op", "open"))
        self.morph_shape_var = tk.StringVar(value=saved.get("morph_shape", "rect"))
        self.morph_iter_var = tk.IntVar(value=int(saved.get("morph_iter", 1)))
        self.roi_margin_var = tk.IntVar(value=saved["roi_margin"])
        self.solidity_var = tk.IntVar(value=saved["solidity"])
        self.min_area_var = tk.StringVar(value=str(saved["min_area"]))
        self.max_area_var = tk.StringVar(value=str(saved["max_area"]))
        self.exclude_tolerance_var = tk.IntVar(value=int(saved.get("exclude_tolerance", 25)))
        self.canny_enabled_var = tk.BooleanVar(value=saved.get("canny_enabled", False))
        self.canny_low_var = tk.IntVar(value=int(saved.get("canny_low", 50)))
        self.canny_high_var = tk.IntVar(value=int(saved.get("canny_high", 150)))
        self.canny_dilate_var = tk.IntVar(value=int(saved.get("canny_dilate", 2)))

        self.section_contour = CollapsibleSection(
            sections_frame, "Contour / binarization settings", start_expanded=True)
        self.section_contour.pack(fill="x", pady=(0, 6))
        self._build_contour_controls(self.section_contour.content)

        self.section_edges = CollapsibleSection(
            sections_frame, "Edge detection & morphology", start_expanded=False)
        self.section_edges.pack(fill="x", pady=(0, 6))
        self._build_edge_morph_controls(self.section_edges.content)

        # ------------------------------------------------------------------
        # Session browser — bottom, always visible, scrollable
        # ------------------------------------------------------------------
        self._build_session_browser()

        # Show/hide manual settings + color controls based on initial mode
        self._on_mode_toggle(confirm=False)

        self.update_frame()

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _on_mode_toggle(self, confirm=True):
        switching_to_manual = not self.use_ai_mode.get()

        if confirm and switching_to_manual:
            proceed = messagebox.askyesno(
                "Switch to manual mode?",
                "Manual mode uses basic threshold/contour detection instead of the "
                "trained AI model. It's generally less accurate and needs careful "
                "slider tuning per product/lighting.\n\nAre you sure you want to "
                "switch to manual mode?"
            )
            if not proceed:
                self.use_ai_mode.set(True)
                return

        if self.use_ai_mode.get():
            self.manual_frame.pack_forget()
            self.manual_settings_outer.pack_forget()
        else:
            self.manual_frame.pack(fill="x", after=self.live_label.master)
            self.manual_settings_outer.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.save_current_settings()

    # ------------------------------------------------------------------
    # Control builders
    # ------------------------------------------------------------------

    def _build_contour_controls(self, parent):
        row = tk.Frame(parent, bg=theme.BG_PANEL)
        row.pack(fill="x")
        self._make_slider(row, "Threshold", self.thresh_var, 0, 255, 0)
        self._make_slider(row, "Blur kernel", self.blur_var, 1, 25, 1)
        self._make_slider(row, "Blur sigma", self.blur_sigma_var, 0, 20, 2)
        self._make_slider(row, "ROI edge crop (%)", self.roi_margin_var, 0, 40, 3)

        row2 = tk.Frame(parent, bg=theme.BG_PANEL)
        row2.pack(fill="x")
        self._make_slider(row2, "Min solidity (%)", self.solidity_var, 0, 100, 0)
        self._make_entry(row2, "Min area", self.min_area_var, 1)
        self._make_entry(row2, "Max area", self.max_area_var, 2)

        inv_chk = tk.Checkbutton(row2, text="Invert mask", variable=self.invert_var,
                                  bg=theme.BG_PANEL, fg=theme.CREAM,
                                  selectcolor=theme.BG_DARK, font=theme.FONT_LABEL,
                                  command=self.save_current_settings)
        inv_chk.grid(row=0, column=6, padx=10)

    def _build_edge_morph_controls(self, parent):
        row = tk.Frame(parent, bg=theme.BG_PANEL)
        row.pack(fill="x")

        canny_chk = tk.Checkbutton(
            row, text="Enable Canny edge boost", variable=self.canny_enabled_var,
            bg=theme.BG_PANEL, fg=theme.CREAM, selectcolor=theme.BG_DARK,
            font=theme.FONT_LABEL, command=self.save_current_settings)
        canny_chk.grid(row=0, column=0, padx=8, pady=8, sticky="w")

        self._make_slider(row, "Canny low", self.canny_low_var, 0, 255, 1)
        self._make_slider(row, "Canny high", self.canny_high_var, 0, 255, 2)
        self._make_slider(row, "Edge dilate (px)", self.canny_dilate_var, 0, 15, 3)

        row2 = tk.Frame(parent, bg=theme.BG_PANEL)
        row2.pack(fill="x")

        tk.Label(row2, text="Morph operation", bg=theme.BG_PANEL, fg=theme.CREAM,
                 font=theme.FONT_LABEL).grid(row=0, column=0, padx=8, pady=(8, 0), sticky="s")
        morph_op_combo = ttk.Combobox(row2, textvariable=self.morph_op_var, width=10,
                                       values=["none", "open", "close", "dilate", "erode",
                                               "gradient"], state="readonly")
        morph_op_combo.grid(row=1, column=0, padx=8, pady=(0, 8))
        morph_op_combo.bind("<<ComboboxSelected>>", lambda e: self.save_current_settings())

        tk.Label(row2, text="Kernel shape", bg=theme.BG_PANEL, fg=theme.CREAM,
                 font=theme.FONT_LABEL).grid(row=0, column=1, padx=8, pady=(8, 0), sticky="s")
        morph_shape_combo = ttk.Combobox(row2, textvariable=self.morph_shape_var, width=10,
                                          values=["rect", "ellipse", "cross"], state="readonly")
        morph_shape_combo.grid(row=1, column=1, padx=8, pady=(0, 8))
        morph_shape_combo.bind("<<ComboboxSelected>>", lambda e: self.save_current_settings())

        self._make_slider(row2, "Morph kernel (px)", self.morph_var, 0, 15, 2)
        self._make_slider(row2, "Morph iterations", self.morph_iter_var, 1, 10, 3)

    def _make_slider(self, parent, text, var, lo, hi, col, tip=None):
        lbl = tk.Label(parent, text=text, bg=theme.BG_PANEL, fg=theme.CREAM,
                        font=theme.FONT_LABEL)
        lbl.grid(row=0, column=col*2, padx=8, pady=(8, 0), sticky="s")
        s = tk.Scale(parent, from_=lo, to=hi, orient="horizontal", variable=var,
                     bg=theme.BG_PANEL, fg=theme.CREAM, troughcolor=theme.BG_DARK,
                     highlightthickness=0, length=160)
        s.grid(row=1, column=col*2, padx=8, pady=(0, 8))
        s.bind("<ButtonRelease-1>", lambda e: self.save_current_settings())
        if tip:
            Tooltip(lbl, tip)
            Tooltip(s, tip)

    def _make_entry(self, parent, text, var, col, tip=None):
        lbl = tk.Label(parent, text=text, bg=theme.BG_PANEL, fg=theme.CREAM,
                        font=theme.FONT_LABEL)
        lbl.grid(row=0, column=col*2, padx=8, pady=(8, 0), sticky="s")
        e = tk.Entry(parent, textvariable=var, width=10,
                     bg=theme.BG_DARK, fg=theme.CREAM,
                     insertbackground=theme.CREAM, font=theme.FONT_LABEL)
        e.grid(row=1, column=col*2, padx=8, pady=(0, 8))
        e.bind("<Return>", lambda ev: self.save_current_settings())
        e.bind("<FocusOut>", lambda ev: self.save_current_settings())
        if tip:
            Tooltip(lbl, tip)
            Tooltip(e, tip)

    # ------------------------------------------------------------------
    # Session browser
    # ------------------------------------------------------------------

    def _build_session_browser(self):
        browser_frame = tk.Frame(self, bg=theme.BG_DARK)
        browser_frame.pack(fill="both", expand=False, padx=10, pady=(0, 10))

        tk.Label(browser_frame, text=f"Recorded sessions — {self.product_name}",
                 bg=theme.BG_DARK, fg=theme.ACCENT_ORANGE,
                 font=theme.FONT_LABEL).pack(anchor="w")

        list_container = tk.Frame(browser_frame, bg=theme.BG_DARK)
        list_container.pack(fill="both", expand=True)

        self.session_listbox = tk.Listbox(
            list_container, height=6, bg=theme.BG_PANEL, fg=theme.CREAM,
            selectbackground=theme.ACCENT_ORANGE, selectforeground=theme.BG_DARK,
            font=theme.FONT_LABEL, highlightthickness=0, bd=0
        )
        self.session_listbox.pack(side="left", fill="both", expand=True)
        self.session_listbox.bind("<Double-Button-1>", lambda e: self.open_selected_session())

        sb = tk.Scrollbar(list_container, orient="vertical", command=self.session_listbox.yview)
        sb.pack(side="right", fill="y")
        self.session_listbox.configure(yscrollcommand=sb.set)

        action_row = tk.Frame(browser_frame, bg=theme.BG_DARK)
        action_row.pack(pady=6)

        open_btn = tk.Button(action_row, text="Open in File Explorer",
                              command=self.open_selected_session)
        theme.style_button(open_btn)
        open_btn.pack(side="left", padx=5)

        refresh_btn = tk.Button(action_row, text="Refresh list",
                                 command=self.refresh_session_list)
        theme.style_button(refresh_btn)
        refresh_btn.pack(side="left", padx=5)

        self.refresh_session_list()

    def refresh_session_list(self):
        self.session_listbox.delete(0, tk.END)
        sessions = du.list_sessions(self.product_name)
        for s in sessions:
            self.session_listbox.insert(tk.END, s)

    def open_selected_session(self):
        selection = self.session_listbox.curselection()
        if not selection:
            messagebox.showinfo("No selection", "Select a session from the list first.")
            return
        session_name = self.session_listbox.get(selection[0])
        session_path = os.path.join(du.DATA_DIR, self.product_name, session_name)

        if not os.path.isdir(session_path):
            messagebox.showerror("Not found", f"Folder no longer exists:\n{session_path}")
            self.refresh_session_list()
            return

        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(session_path)
            elif system == "Darwin":
                subprocess.Popen(["open", session_path])
            else:
                subprocess.Popen(["xdg-open", session_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{e}")

    # ------------------------------------------------------------------
    # Color filtering (manual mode only)
    # ------------------------------------------------------------------

    def _active_color_list(self):
        return self.include_colors if self.color_mode_var.get() == "include" else self.exclude_colors

    def _update_color_count_label(self):
        mode = self.color_mode_var.get()
        n = len(self._active_color_list())
        label = "Included" if mode == "include" else "Excluded"
        self.excl_count_label.configure(text=f"{label} colors: {n}")

    def _on_color_mode_change(self):
        self._update_color_count_label()
        self.save_current_settings()

    def open_color_picker(self):
        for _ in range(2):
            self.cap.grab()
        ret, frame = self.cap.retrieve()
        if not ret:
            return

        mode = self.color_mode_var.get()

        def on_sample(h, s, v):
            self._active_color_list().append([h, s, v])
            self._update_color_count_label()
            self.save_current_settings()

        mode_label = "include" if mode == "include" else "exclude"
        ColorPickerWindow(self, frame.copy(), on_sample, mode_label=mode_label)

    def open_color_manager(self):
        mode = self.color_mode_var.get()
        label = "Include" if mode == "include" else "Exclude"

        win = tk.Toplevel(self)
        win.title(f"Manage {label.lower()} colors")
        theme.style_window(win)

        tk.Label(win, text=f"{label} colors — click Remove to delete a sample.",
                 bg=theme.BG_DARK, fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=(10, 5))

        list_frame = tk.Frame(win, bg=theme.BG_DARK)
        list_frame.pack(padx=10, pady=5, fill="both", expand=True)

        def refresh():
            for w in list_frame.winfo_children():
                w.destroy()

            color_list = self._active_color_list()

            if not color_list:
                tk.Label(list_frame, text="No colors sampled yet.", bg=theme.BG_DARK,
                         fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=10)
                return

            for i, (h, s, v) in enumerate(color_list):
                row = tk.Frame(list_frame, bg=theme.BG_DARK)
                row.pack(fill="x", pady=2)

                hsv_pixel = np.uint8([[[h, s, v]]])
                bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
                hex_color = "#%02x%02x%02x" % (int(bgr_pixel[2]), int(bgr_pixel[1]), int(bgr_pixel[0]))

                swatch = tk.Label(row, bg=hex_color, width=4, height=1,
                                   relief="solid", bd=1)
                swatch.pack(side="left", padx=(0, 8))

                tk.Label(row, text=f"H:{h}  S:{s}  V:{v}", bg=theme.BG_DARK,
                         fg=theme.CREAM, font=("Consolas", 9)).pack(side="left")

                def remove(idx=i):
                    del self._active_color_list()[idx]
                    self._update_color_count_label()
                    self.save_current_settings()
                    refresh()

                rm_btn = tk.Button(row, text="Remove", command=remove)
                theme.style_button(rm_btn)
                rm_btn.pack(side="right", padx=5)

        refresh()

        close_btn = tk.Button(win, text="Done", command=win.destroy)
        theme.style_button(close_btn)
        close_btn.pack(pady=10)

    def clear_colors(self):
        if self.color_mode_var.get() == "include":
            self.include_colors = []
        else:
            self.exclude_colors = []
        self._update_color_count_label()
        self.save_current_settings()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _get_int_var(self, string_var, fallback):
        try:
            return int(string_var.get())
        except (ValueError, TypeError):
            string_var.set(str(fallback))
            return fallback

    def save_current_settings(self):
        settings = {
            "threshold": self.thresh_var.get(),
            "blur": self.blur_var.get(),
            "blur_sigma": self.blur_sigma_var.get(),
            "invert": self.invert_var.get(),
            "margin": self.margin_var.get(),
            "morph": self.morph_var.get(),
            "morph_op": self.morph_op_var.get(),
            "morph_shape": self.morph_shape_var.get(),
            "morph_iter": self.morph_iter_var.get(),
            "min_area": self._get_int_var(self.min_area_var, 500),
            "max_area": self._get_int_var(self.max_area_var, 200000),
            "roi_margin": self.roi_margin_var.get(),
            "solidity": self.solidity_var.get(),
            "ai_conf": self.ai_conf_var.get(),
            "exclude_colors": self.exclude_colors,
            "include_colors": self.include_colors,
            "color_mode": self.color_mode_var.get(),
            "exclude_tolerance": self.exclude_tolerance_var.get(),
            "canny_enabled": self.canny_enabled_var.get(),
            "canny_low": self.canny_low_var.get(),
            "canny_high": self.canny_high_var.get(),
            "canny_dilate": self.canny_dilate_var.get(),
            "use_ai_mode": self.use_ai_mode.get(),
        }
        save_settings(self.product_name, settings)
        self.status_label.configure(text=f"Settings saved for {self.product_name}.")

    def reset_to_default(self):
        d = self.default_snapshot
        self.thresh_var.set(d["threshold"])
        self.blur_var.set(d["blur"])
        self.blur_sigma_var.set(int(d.get("blur_sigma", 0)))
        self.invert_var.set(d["invert"])
        self.margin_var.set(d["margin"])
        self.morph_var.set(d["morph"])
        self.morph_op_var.set(d.get("morph_op", "open"))
        self.morph_shape_var.set(d.get("morph_shape", "rect"))
        self.morph_iter_var.set(int(d.get("morph_iter", 1)))
        self.roi_margin_var.set(d["roi_margin"])
        self.solidity_var.set(d["solidity"])
        self.min_area_var.set(str(d["min_area"]))
        self.max_area_var.set(str(d["max_area"]))
        self.ai_conf_var.set(int(d.get("ai_conf", 50)))
        self.exclude_colors = list(d.get("exclude_colors", []))
        self.include_colors = list(d.get("include_colors", []))
        self.color_mode_var.set(d.get("color_mode", "exclude"))
        self.exclude_tolerance_var.set(int(d.get("exclude_tolerance", 25)))
        self.canny_enabled_var.set(d.get("canny_enabled", False))
        self.canny_low_var.set(int(d.get("canny_low", 50)))
        self.canny_high_var.set(int(d.get("canny_high", 150)))
        self.canny_dilate_var.set(int(d.get("canny_dilate", 2)))
        self._update_color_count_label()
        self.save_current_settings()
        self.status_label.configure(text="Reset to default values.")

    # ------------------------------------------------------------------
    # AI mode: polygon detection directly from the model
    # ------------------------------------------------------------------

    def compute_polygon_ai(self, frame):
        """Runs the detector and returns (polygon, bbox) for the single
        highest-confidence detection, or (None, None) if nothing found.
        polygon is an (N,2) array of pixel coords in ORIGINAL frame space,
        already grown/shrunk by the margin slider if set."""
        conf_threshold = self.ai_conf_var.get() / 100.0
        self.ai_detector.conf_threshold = conf_threshold
        detections = self.ai_detector.predict(frame)

        if not detections:
            return None, None

        label, conf, bbox, polygon = max(detections, key=lambda d: d[1])

        if polygon is None or len(polygon) < 3:
            return None, bbox

        margin = self.margin_var.get()
        if margin != 0:
            pts = np.array(polygon, dtype=np.float32)
            centroid = pts.mean(axis=0)
            directions = pts - centroid
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms[norms == 0] = 1
            pts = pts + (directions / norms) * margin
            polygon = pts

        return polygon, bbox

    # ------------------------------------------------------------------
    # Manual mode: threshold/contour bbox (legacy)
    # ------------------------------------------------------------------

    def _apply_morphology(self, mask):
        op = self.morph_op_var.get()
        size = self.morph_var.get()
        if op == "none" or size <= 0:
            return mask

        shape_map = {
            "rect": cv2.MORPH_RECT,
            "ellipse": cv2.MORPH_ELLIPSE,
            "cross": cv2.MORPH_CROSS,
        }
        shape = shape_map.get(self.morph_shape_var.get(), cv2.MORPH_RECT)
        kernel = cv2.getStructuringElement(shape, (size, size))
        iterations = self.morph_iter_var.get()

        op_map = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "dilate": cv2.MORPH_DILATE,
            "erode": cv2.MORPH_ERODE,
            "gradient": cv2.MORPH_GRADIENT,
        }
        cv_op = op_map.get(op)
        if cv_op is None:
            return mask

        return cv2.morphologyEx(mask, cv_op, kernel, iterations=iterations)

    def _binarize_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k = self.blur_var.get()
        if k % 2 == 0:
            k += 1
        sigma = self.blur_sigma_var.get()
        gray_blur = cv2.GaussianBlur(gray, (k, k), sigma)
        _, mask = cv2.threshold(gray_blur, self.thresh_var.get(), 255, cv2.THRESH_BINARY)
        if self.invert_var.get():
            mask = cv2.bitwise_not(mask)

        if self.canny_enabled_var.get():
            edges = cv2.Canny(gray_blur, self.canny_low_var.get(), self.canny_high_var.get())
            d = self.canny_dilate_var.get()
            if d > 0:
                edge_kernel = np.ones((d, d), np.uint8)
                edges = cv2.dilate(edges, edge_kernel, iterations=1)
            mask = cv2.bitwise_or(mask, edges)

        mode = self.color_mode_var.get()
        tol = self.exclude_tolerance_var.get()

        if mode == "include" and self.include_colors:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            keep_mask = np.zeros(mask.shape, dtype=np.uint8)
            for h, s, v in self.include_colors:
                lower = np.array([max(0, h - tol), max(0, s - tol), max(0, v - tol)])
                upper = np.array([min(179, h + tol), min(255, s + tol), min(255, v + tol)])
                color_match = cv2.inRange(hsv, lower, upper)
                keep_mask = cv2.bitwise_or(keep_mask, color_match)
            mask = cv2.bitwise_and(mask, keep_mask)

        elif mode == "exclude" and self.exclude_colors:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            exclusion_mask = np.zeros(mask.shape, dtype=np.uint8)
            for h, s, v in self.exclude_colors:
                lower = np.array([max(0, h - tol), max(0, s - tol), max(0, v - tol)])
                upper = np.array([min(179, h + tol), min(255, s + tol), min(255, v + tol)])
                color_match = cv2.inRange(hsv, lower, upper)
                exclusion_mask = cv2.bitwise_or(exclusion_mask, color_match)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclusion_mask))

        mask = self._apply_morphology(mask)
        return mask

    def compute_mask_and_bbox_contour(self, frame):
        h_img, w_img = frame.shape[:2]
        mask = self._binarize_frame(frame)

        roi_pct = self.roi_margin_var.get()
        if roi_pct > 0:
            mx = int(w_img * roi_pct / 100)
            my = int(h_img * roi_pct / 100)
            roi_mask = np.zeros_like(mask)
            roi_mask[my:h_img - my, mx:w_img - mx] = 255
            mask = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self._get_int_var(self.min_area_var, 500)
        max_area = self._get_int_var(self.max_area_var, 200000)
        min_solidity = self.solidity_var.get() / 100.0

        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            if min_solidity > 0:
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                if (area / hull_area) < min_solidity:
                    continue
            if area > best_area:
                best_area = area
                best = c

        polygon = None
        bbox = None
        if best is not None:
            x, y, w, h = cv2.boundingRect(best)
            margin = self.margin_var.get()
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = max(1, min(w_img - x, w + 2 * margin))
            h = max(1, min(h_img - y, h + 2 * margin))
            bbox = (x, y, w, h)

            epsilon = 0.004 * cv2.arcLength(best, True)
            approx = cv2.approxPolyDP(best, epsilon, True)
            if len(approx) >= 3:
                polygon = approx.reshape(-1, 2)

        return mask, polygon, bbox

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _draw_polygon(self, display, polygon, color=(0, 122, 255), alpha=0.35):
        pts = np.array(polygon, dtype=np.int32)
        if pts.shape[0] < 3:
            return
        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, dst=display)
        cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)

    def update_frame(self):
        for _ in range(2):
            self.cap.grab()
        ret, frame = self.cap.retrieve()

        if ret:
            h_img, w_img = frame.shape[:2]
            polygon = None
            bbox = None

            if self.use_ai_mode.get() and self.ai_detector.loaded:
                polygon, bbox = self.compute_polygon_ai(frame)
            else:
                _, polygon, bbox = self.compute_mask_and_bbox_contour(frame)

            live_draw = frame.copy()
            mask_draw = np.zeros_like(frame)

            if polygon is not None:
                self._draw_polygon(live_draw, polygon)
                self._draw_polygon(mask_draw, polygon, color=(0, 255, 0))
            elif bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(live_draw, (x, y), (x + w, y + h), (0, 122, 255), 2)
                cv2.rectangle(mask_draw, (x, y), (x + w, y + h), (0, 122, 255), 2)

            if self.recording and polygon is not None:
                self.save_sample(frame, polygon)

            self._show(self.live_label, live_draw)
            self._show(self.mask_label, mask_draw)

        self.after(30, self.update_frame)

    def _show(self, label, cv_img):
        cv_img = cv2.resize(cv_img, (400, 300))
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        label.imgtk = imgtk
        label.configure(image=imgtk, width=400, height=300)

    def toggle_record(self):
        self.recording = not self.recording
        if self.recording:
            self.session_path = du.new_session_path(self.product_name)
            self.frame_idx = 0
            self.record_btn.configure(text="■ Stop recording", bg=theme.ACCENT_YELLOW)
            self.status_label.configure(text=f"Recording to {os.path.basename(self.session_path)}")
        else:
            self.record_btn.configure(text="● Record training data", bg=theme.ACCENT_ORANGE)
            self.status_label.configure(text=f"Saved {self.frame_idx} samples.")
            self.refresh_session_list()

    def save_sample(self, frame, polygon):
        """Saves the full frame + a YOLO-segmentation label (normalized
        polygon), using the polygon already computed by whichever mode
        (AI or manual) is currently active."""
        h_img, w_img = frame.shape[:2]

        pts = np.array(polygon, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)

        if len(pts) < 3:
            return

        norm_coords = []
        for px, py in pts:
            norm_coords.append(px / w_img)
            norm_coords.append(py / h_img)

        fname = f"{self.frame_idx:05d}"
        img_path = os.path.join(self.session_path, "images", fname + ".jpg")
        lbl_path = os.path.join(self.session_path, "labels", fname + ".txt")

        cv2.imwrite(img_path, frame)
        with open(lbl_path, "w") as f:
            coords_str = " ".join(f"{v:.6f}" for v in norm_coords)
            f.write(f"0 {coords_str}\n")

        self.frame_idx += 1

    def on_close(self):
        self.save_current_settings()
        self.cap.release()
        self.destroy()
