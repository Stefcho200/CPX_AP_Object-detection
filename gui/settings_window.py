import cv2
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
from core.camera_utils import list_available_cameras, probe_ip_camera
from core.config_utils import (
    get_camera_index, get_camera_source, set_camera_source, add_known_ip
)
from core.i18n import t, tf, on_language_change
import theme
from gui.animation import DropdownAnimationMixin


class SettingsWindow(tk.Toplevel, DropdownAnimationMixin):
    def __init__(self, master, anchor_widget=None):
        super().__init__(master)
        self.title(t("settings_title"))
        theme.style_window(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.title_label = tk.Label(self, text=t("settings_title"), bg=theme.BG_DARK,
                                     fg=theme.ACCENT_ORANGE, font=theme.FONT_TITLE)
        self.title_label.pack(pady=15)

        self.camera_label = tk.Label(self, text=t("settings_camera_title"), bg=theme.BG_DARK,
                                      fg=theme.CREAM, font=theme.FONT_LABEL)
        self.camera_label.pack(pady=5)

        self.status_label = tk.Label(self, text=t("scanning_cameras"), bg=theme.BG_DARK,
                                      fg=theme.ACCENT_YELLOW, font=theme.FONT_LABEL)
        self.status_label.pack(pady=5)

        self.grid_frame = tk.Frame(self, bg=theme.BG_DARK)
        self.grid_frame.pack(pady=10)

        btn_row = tk.Frame(self, bg=theme.BG_DARK)
        btn_row.pack(pady=10)

        self.auto_btn = tk.Button(btn_row, text=t("use_auto_detect"), command=self.use_auto)
        theme.style_button(self.auto_btn)
        self.auto_btn.pack(side="left", padx=5)

        self.ip_btn = tk.Button(btn_row, text=t("ip_camera_btn") if _has_key("ip_camera_btn") else "IP Camera",
                                 command=self.add_ip_camera)
        theme.style_button(self.ip_btn)
        self.ip_btn.pack(side="left", padx=5)

        self.current_label = tk.Label(self, text=self._current_setting_text(), bg=theme.BG_DARK,
                                       fg=theme.CREAM, font=theme.FONT_LABEL)
        self.current_label.pack(pady=5)

        self.close_btn = tk.Button(self, text=t("close"), command=self.on_close)
        theme.style_button(self.close_btn)
        self.close_btn.pack(pady=10)

        self._unsub_lang = on_language_change(self._on_language_changed)

        self.animate_dropdown_open(width=560, height=480, anchor_widget=anchor_widget)

        self.after(100, self.scan_and_display)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_setting_text(self):
        source = get_camera_source()
        if source["type"] == "ip" and source["value"]:
            return f"Current: IP Camera ({source['value']})"
        if source["value"] is None:
            return t("current_setting_auto")
        return tf("current_setting_index", i=source["value"])

    def _on_language_changed(self, lang_code):
        self.title(t("settings_title"))
        self.title_label.configure(text=t("settings_title"))
        self.camera_label.configure(text=t("settings_camera_title"))
        self.auto_btn.configure(text=t("use_auto_detect"))
        self.close_btn.configure(text=t("close"))
        self.current_label.configure(text=self._current_setting_text())
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.scan_and_display()

    def scan_and_display(self):
        cameras = list_available_cameras()
        self.status_label.configure(text=tf("found_camera_slots", n=len(cameras)))

        col = 0
        for cam in cameras:
            frame_widget = tk.Frame(self.grid_frame, bg=theme.BG_PANEL, padx=8, pady=8)
            frame_widget.grid(row=0, column=col, padx=8)

            if cam["preview"] is not None:
                img = cv2.resize(cam["preview"], (160, 120))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
                img_label = tk.Label(frame_widget, image=imgtk, bg=theme.BG_PANEL)
                img_label.imgtk = imgtk
                img_label.pack()
            else:
                tk.Label(frame_widget, text=t("no_signal"), bg=theme.BG_PANEL,
                          fg=theme.CREAM, width=20, height=7,
                          font=theme.FONT_LABEL).pack()

            status_text = t("usable") if cam["usable"] else t("unusable")
            status_color = theme.ACCENT_ORANGE if cam["usable"] else "#888888"

            if cam["type"] == "local":
                label_text = tf("camera_index_label", i=cam['index'])
            else:
                label_text = f"IP: {cam['ip']}"

            tk.Label(frame_widget, text=label_text,
                      bg=theme.BG_PANEL, fg=theme.CREAM,
                      font=theme.FONT_LABEL).pack(pady=(5, 0))
            tk.Label(frame_widget, text=status_text, bg=theme.BG_PANEL,
                      fg=status_color, font=theme.FONT_LABEL).pack()

            if cam["type"] == "local":
                select_cmd = lambda i=cam["index"]: self.select_local_camera(i)
            else:
                select_cmd = lambda ip=cam["ip"]: self.select_ip_camera(ip)

            select_btn = tk.Button(
                frame_widget, text=t("select_btn"),
                command=select_cmd,
                state="normal" if cam["usable"] else "disabled"
            )
            theme.style_button(select_btn)
            select_btn.pack(pady=(8, 0))

            col += 1

    # ------------------------------------------------------------------
    # Selection actions
    # ------------------------------------------------------------------

    def select_local_camera(self, index):
        set_camera_source("index", index)
        self.current_label.configure(text=self._current_setting_text())
        self.status_label.configure(text=tf("camera_selected", i=index))

    def select_ip_camera(self, ip):
        set_camera_source("ip", ip)
        add_known_ip(ip)
        self.current_label.configure(text=self._current_setting_text())
        self.status_label.configure(text=f"IP camera selected: {ip}")

    def use_auto(self):
        set_camera_source("index", None)
        self.current_label.configure(text=self._current_setting_text())
        self.status_label.configure(text=t("reverted_auto"))

    def add_ip_camera(self):
        ip = simpledialog.askstring("IP Camera", "Enter camera IP address:", parent=self)
        if not ip:
            return
        ip = ip.strip()

        self.status_label.configure(text=f"Testing connection to {ip}...")
        self.update_idletasks()

        result = probe_ip_camera(ip)

        if result["usable"]:
            add_known_ip(ip)
            set_camera_source("ip", ip)
            self.current_label.configure(text=self._current_setting_text())
            self.status_label.configure(text=f"Connected to {ip} successfully")
            for widget in self.grid_frame.winfo_children():
                widget.destroy()
            self.scan_and_display()
        else:
            self.status_label.configure(
                text=f"Could not connect to {ip} - check the IP and try again",
                fg=theme.ACCENT_YELLOW if hasattr(theme, "ACCENT_YELLOW") else theme.CREAM
            )

    def on_close(self):
        self.destroy()


def _has_key(key):
    """Guard against missing i18n keys until translations are added."""
    try:
        val = t(key)
        return val != key
    except Exception:
        return False
