import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from core import cleanup_utils as cu
import theme


class CleanupViewer(tk.Toplevel):
    """Frame-by-frame player over an arbitrary set of recorded sessions,
    walking image+label pairs directly off disk (never generating a
    video file or copy of any kind). Supports play/pause, step, speed,
    and flagging bad frames for later batch deletion.

    Sessions play back-to-back in order, but the current product/session
    name is always shown, so distinct sessions remain visually separate
    even though there's no hard "stop" between them."""

    SPEED_OPTIONS = {
        "0.25x": 480, "0.5x": 240, "1x": 120, "2x": 60, "4x": 30,
    }

    def __init__(self, master, frames, title="Training Data Viewer", review_mode=False):
        super().__init__(master)
        self.title(title)
        theme.style_window(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.geometry("760x760" if review_mode else "760x680")
        self.minsize(760, 760 if review_mode else 680)
        self.review_mode = review_mode


        if not frames:
            tk.Label(self, text="No frames found for this selection.",
                     bg=theme.BG_DARK, fg=theme.CREAM, font=theme.FONT_LABEL).pack(pady=40)
            close_btn = tk.Button(self, text="Close", command=self.destroy)
            theme.style_button(close_btn)
            close_btn.pack(pady=10)
            return

        self.frames = frames
        self.index = 0
        self.playing = False
        self.speed_ms = self.SPEED_OPTIONS["1x"]
        self.flags = cu.load_flags()

        # --- Video display ---
        self.canvas_label = tk.Label(self, bg=theme.BG_PANEL, width=640, height=480)
        self.canvas_label.pack(padx=10, pady=10)
        self.canvas_label.pack_propagate(False)

        # --- Info row ---
        self.info_label = tk.Label(self, text="", bg=theme.BG_DARK, fg=theme.CREAM,
                                    font=theme.FONT_LABEL)
        self.info_label.pack(pady=(0, 5))

        self.flag_status_label = tk.Label(self, text="", bg=theme.BG_DARK,
                                           fg=theme.ACCENT_YELLOW, font=theme.FONT_LABEL)
        self.flag_status_label.pack(pady=(0, 5))

        # --- Transport controls ---
        controls = tk.Frame(self, bg=theme.BG_DARK)
        controls.pack(pady=5)

        def ctrl_btn(parent, text, cmd, width=4):
            b = tk.Button(parent, text=text, command=cmd, width=width)
            theme.style_button(b)
            b.pack(side="left", padx=3)
            return b

        ctrl_btn(controls, "|<<", self.jump_prev_session)
        ctrl_btn(controls, "<<", self.step_back)
        self.play_btn = ctrl_btn(controls, "▶ Play", self.toggle_play, width=8)
        ctrl_btn(controls, ">>", self.step_forward)
        ctrl_btn(controls, ">>|", self.jump_next_session)

        speed_frame = tk.Frame(self, bg=theme.BG_DARK)
        speed_frame.pack(pady=5)
        tk.Label(speed_frame, text="Speed:", bg=theme.BG_DARK, fg=theme.CREAM,
                 font=theme.FONT_LABEL).pack(side="left", padx=5)
        self.speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var, width=6,
                                    values=list(self.SPEED_OPTIONS.keys()), state="readonly")
        speed_combo.pack(side="left")
        speed_combo.bind("<<ComboboxSelected>>", self.on_speed_change)

        # --- Scrub slider ---
        self.scrub_var = tk.IntVar(value=0)
        self.scrub_slider = tk.Scale(
            self, from_=0, to=len(self.frames) - 1, orient="horizontal",
            variable=self.scrub_var, bg=theme.BG_DARK, fg=theme.CREAM,
            troughcolor=theme.BG_PANEL, highlightthickness=0, length=700,
            showvalue=False, command=self.on_scrub
        )
        self.scrub_slider.pack(pady=5)

        # --- Flag + delete controls ---
        flag_frame = tk.Frame(self, bg=theme.BG_DARK)
        flag_frame.pack(pady=8)

        self.flag_btn = tk.Button(flag_frame, text="🚩 Flag this frame",
                                   command=self.toggle_current_flag)
        theme.style_button(self.flag_btn)
        self.flag_btn.pack(side="left", padx=5)

        self.flag_count_label = tk.Label(flag_frame, text="", bg=theme.BG_DARK,
                                          fg=theme.CREAM, font=theme.FONT_LABEL)
        self.flag_count_label.pack(side="left", padx=10)

        delete_btn = tk.Button(flag_frame, text="🗑 Delete all flagged frames",
                                command=self.delete_flagged)
        theme.style_button(delete_btn)
        delete_btn.pack(side="left", padx=5)

        if self.review_mode:
            banner = tk.Label(
                self, text="⚠ REVIEWING STAGED RE-LABEL — nothing is saved to your "
                           "dataset yet. Commit to make it permanent, or discard to "
                           "throw the new labels away and keep the originals.",
                bg="#5a3d00", fg=theme.ACCENT_YELLOW, font=("Consolas", 9, "bold"),
                wraplength=700, justify="center")
            banner.pack(fill="x", padx=5, pady=(0, 5))

            commit_frame = tk.Frame(self, bg=theme.BG_DARK)
            commit_frame.pack(pady=(0, 5))

            commit_btn = tk.Button(commit_frame, text="✓ Commit staged re-labels (permanent)",
                                    command=self.commit_staging)
            theme.style_button(commit_btn)
            commit_btn.pack(side="left", padx=5)

            discard_btn = tk.Button(commit_frame, text="✗ Discard staged re-labels",
                                     command=self.discard_staging)
            theme.style_button(discard_btn)
            discard_btn.pack(side="left", padx=5)

        close_btn = tk.Button(self, text="Close", command=self.on_close)
        theme.style_button(close_btn)
        close_btn.pack(pady=10)

        self._after_id = None
        self.show_frame(0)


    # ------------------------------------------------------------------
    # Frame rendering
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
                x = coords[i] * w_img
                y = coords[i + 1] * h_img
                pts.append([x, y])
            return np.array(pts, dtype=np.int32)
        except Exception:
            return None

    def show_frame(self, index):
        index = max(0, min(len(self.frames) - 1, index))
        self.index = index
        info = self.frames[index]

        img = cv2.imread(info["image_path"])
        if img is None:
            self.info_label.configure(text=f"[Missing/unreadable file: {info['image_path']}]")
            return

        h_img, w_img = img.shape[:2]
        label_path = info.get("staged_label_path", info.get("label_path"))
        polygon = self._load_polygon(label_path, w_img, h_img)


        display = img.copy()
        if polygon is not None and len(polygon) >= 3:
            overlay = display.copy()
            cv2.fillPoly(overlay, [polygon], (0, 200, 0))
            cv2.addWeighted(overlay, 0.35, display, 0.65, 0, dst=display)
            cv2.polylines(display, [polygon], True, (0, 255, 0), 2)
        else:
            cv2.putText(display, "NO VALID LABEL", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        disp = cv2.resize(display, (640, 480))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas_label.imgtk = imgtk
        self.canvas_label.configure(image=imgtk)

        fname = os.path.basename(info["image_path"])
        self.info_label.configure(
            text=f"[{index + 1}/{len(self.frames)}]  "
                 f"{info['product']} / {info['session']} / {fname}")

        is_flagged = info["image_path"] in self.flags
        self.flag_btn.configure(
            text="🚩 Unflag this frame" if is_flagged else "🚩 Flag this frame",
            bg="#c0392b" if is_flagged else theme.ACCENT_ORANGE)
        self.flag_count_label.configure(text=f"Flagged total: {len(self.flags)}")

        self.scrub_var.set(index)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def step_forward(self):
        self.show_frame(self.index + 1)

    def step_back(self):
        self.show_frame(self.index - 1)

    def jump_next_session(self):
        cur = self.frames[self.index]
        for i in range(self.index + 1, len(self.frames)):
            if self.frames[i]["session"] != cur["session"] or self.frames[i]["product"] != cur["product"]:
                self.show_frame(i)
                return
        self.show_frame(len(self.frames) - 1)

    def jump_prev_session(self):
        cur = self.frames[self.index]
        # First, move back to start of CURRENT session if we're mid-way through it.
        start_of_current = self.index
        while start_of_current > 0 and \
                self.frames[start_of_current - 1]["session"] == cur["session"] and \
                self.frames[start_of_current - 1]["product"] == cur["product"]:
            start_of_current -= 1

        if start_of_current != self.index:
            self.show_frame(start_of_current)
            return

        # Already at the start of this session — jump to start of the previous one.
        if start_of_current == 0:
            return
        prev_session_end = start_of_current - 1
        prev_info = self.frames[prev_session_end]
        start_of_prev = prev_session_end
        while start_of_prev > 0 and \
                self.frames[start_of_prev - 1]["session"] == prev_info["session"] and \
                self.frames[start_of_prev - 1]["product"] == prev_info["product"]:
            start_of_prev -= 1
        self.show_frame(start_of_prev)

    def on_scrub(self, value):
        self.show_frame(int(value))

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._tick()

    def _tick(self):
        if not self.playing:
            return
        if self.index >= len(self.frames) - 1:
            self.playing = False
            self.play_btn.configure(text="▶ Play")
            return
        self.step_forward()
        self._after_id = self.after(self.speed_ms, self._tick)

    def on_speed_change(self, event=None):
        self.speed_ms = self.SPEED_OPTIONS.get(self.speed_var.get(), 120)

    # ------------------------------------------------------------------
    # Flagging
    # ------------------------------------------------------------------

    def toggle_current_flag(self):
        img_path = self.frames[self.index]["image_path"]
        cu.toggle_flag(img_path, self.flags)
        self.show_frame(self.index)

    def delete_flagged(self):
        if not self.flags:
            messagebox.showinfo("No flags", "No frames are currently flagged.")
            return

        confirm = messagebox.askyesno(
            "Confirm deletion",
            f"Permanently delete {len(self.flags)} flagged frame(s) "
            f"(image + label) from disk?\n\nThis cannot be undone."
        )
        if not confirm:
            return

        deleted = cu.delete_flagged(self.flags)

        # Remove deleted frames from our in-memory list + fix up the
        # scrub slider range, rather than closing the whole viewer.
        self.frames = [f for f in self.frames if os.path.exists(f["image_path"])]
        self.scrub_slider.configure(to=max(0, len(self.frames) - 1))

        messagebox.showinfo("Deleted", f"Deleted {deleted} frame(s).")

        if not self.frames:
            self.destroy()
            return

        self.show_frame(min(self.index, len(self.frames) - 1))


    def commit_staging(self):
        from core import cleanup_utils as cu
        count = len(self.frames)
        confirm = messagebox.askyesno(
            "Confirm commit",
            f"This will permanently overwrite {count} real label(s) in your "
            f"dataset with the staged re-labeled versions.\n\n"
            f"This cannot be undone. Continue?"
        )
        if not confirm:
            return
        committed = cu.commit_staged_relabels()
        messagebox.showinfo("Committed", f"Committed {committed} re-labeled frame(s).")
        self.destroy()

    def discard_staging(self):
        from core import cleanup_utils as cu
        confirm = messagebox.askyesno(
            "Confirm discard",
            "This will throw away all staged re-labeled results. Your original "
            "dataset labels remain completely untouched.\n\nContinue?"
        )
        if not confirm:
            return
        cu.discard_staged_relabels()
        messagebox.showinfo("Discarded", "Staged re-labels discarded.")
        self.destroy()








    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def on_close(self):
        self.playing = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
        self.destroy()
