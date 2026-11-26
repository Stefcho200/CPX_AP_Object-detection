import tkinter as tk
from core import i18n
from core import settings_manager
import theme


class TopBar(tk.Frame):
    """Flag icon (top-left) -> language dropdown, and sun/moon icon
    (top-right) -> theme toggle. Drop into any Toplevel/Tk root via
    TopBar(self).pack(fill='x') as the FIRST packed widget."""

    def __init__(self, parent, on_language_changed=None, on_theme_changed=None):
        super().__init__(parent, bg=theme.BG_DARK)
        self._on_language_changed = on_language_changed
        self._on_theme_changed = on_theme_changed

        self.flag_btn = tk.Button(
            self, text=i18n.LANGUAGES[i18n.get_language()]["flag"],
            command=self._open_language_menu
        )
        theme.style_button(self.flag_btn)
        self.flag_btn.pack(side="left", padx=10, pady=6)

        self.theme_btn = tk.Button(
            self, text=self._theme_icon(), command=self._toggle_theme
        )
        theme.style_button(self.theme_btn)
        self.theme_btn.pack(side="right", padx=10, pady=6)

    def _theme_icon(self):
        return "☀" if theme.get_theme() == "dark" else "🌙"

    def _toggle_theme(self):
        new_theme = "light" if theme.get_theme() == "dark" else "dark"
        theme.apply_theme(new_theme)
        settings_manager.update_setting("theme", new_theme)
        theme.style_button(self.theme_btn)
        theme.style_button(self.flag_btn)
        self.theme_btn.configure(text=self._theme_icon())
        self.configure(bg=theme.BG_DARK)
        if self._on_theme_changed:
            self._on_theme_changed(new_theme)

    def _open_language_menu(self):
        menu = tk.Menu(self, tearoff=0)
        for code, data in i18n.LANGUAGES.items():
            menu.add_command(
                label=f"{data['flag']}  {data['display_name']}",
                command=lambda c=code: self._select_language(c)
            )
        x = self.flag_btn.winfo_rootx()
        y = self.flag_btn.winfo_rooty() + self.flag_btn.winfo_height()
        menu.tk_popup(x, y)

    def _select_language(self, lang_code):
        i18n.set_language(lang_code)
        settings_manager.update_setting("language", lang_code)
        self.flag_btn.configure(text=i18n.LANGUAGES[lang_code]["flag"])
        if self._on_language_changed:
            self._on_language_changed(lang_code)
