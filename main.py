import tkinter as tk
from gui.detect_window import DetectWindow
from gui.training_menu import TrainingMenu
from gui.settings_window import SettingsWindow
from gui.top_bar import TopBar
from core.i18n import t
import theme
from core import settings_manager
settings_manager.apply_saved_settings()


class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Festo CPX Recognizer")
        self.geometry("480x420")
        theme.style_window(self)

        self._open_windows = {}

        TopBar(self, on_language_changed=self._on_language_changed,
               on_theme_changed=self._on_theme_changed).pack(fill="x")

        self.title_label = tk.Label(self, text=t("app_title"), bg=theme.BG_DARK,
                                     fg=theme.CREAM, font=theme.FONT_TITLE)
        self.title_label.pack(pady=30)

        self.detect_btn = tk.Button(self, text=t("menu_detect"), command=self.open_detect)
        theme.style_button(self.detect_btn)
        self.detect_btn.pack(pady=15, ipadx=20)

        self.train_btn = tk.Button(self, text=t("menu_training"), command=self.open_training)
        theme.style_button(self.train_btn)
        self.train_btn.pack(pady=15, ipadx=20)

        self.settings_btn = tk.Button(self, text=t("menu_settings"), command=self.open_settings)
        theme.style_button(self.settings_btn)
        self.settings_btn.pack(pady=15, ipadx=20)

    def _on_language_changed(self, lang_code):
        print(f"[main_menu] language changed to: {lang_code}")  # temp debug line
        self.title_label.configure(text=t("app_title"))
        self.detect_btn.configure(text=t("menu_detect"))
        self.train_btn.configure(text=t("menu_training"))
        self.settings_btn.configure(text=t("menu_settings"))

    def _on_theme_changed(self, theme_name):
        theme.refresh_widget_tree(self)
        self.title_label.configure(fg=theme.CREAM)

    def _open_or_focus(self, key, window_class, *args, **kwargs):
        win = self._open_windows.get(key)
        if win is not None and win.winfo_exists():
            win.lift()
            win.focus_force()
            return

        win = window_class(self, *args, **kwargs)
        self._open_windows[key] = win
        win.bind("<Destroy>", lambda e, k=key: self._open_windows.pop(k, None))

    def open_detect(self):
        self._open_or_focus("detect", DetectWindow)

    def open_training(self):
        self._open_or_focus("training", TrainingMenu, anchor_widget=self.train_btn)

    def open_settings(self):
        self._open_or_focus("settings", SettingsWindow, anchor_widget=self.settings_btn)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    app = MainMenu()
    app.mainloop()
