"""Shared styling. DARK is your original color scheme, unchanged.
LIGHT is Festo's signature palette. Module-level names (BG_DARK,
CREAM, etc.) are live references repointed by apply_theme() - existing
code doing `theme.BG_DARK` keeps working unchanged after a switch.
"""

FONT_TITLE = ("Consolas", 20, "bold")
FONT_BTN   = ("Consolas", 14, "bold")
FONT_LABEL = ("Consolas", 11)

# ----------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------

DARK_PALETTE = {
    "BG_DARK": "#1a1a1a",
    "BG_PANEL": "#3f3f3f",
    "ACCENT_ORANGE": "#ff7a00",
    "ACCENT_YELLOW": "#ffc266",
    "CREAM": "#f2e2c4",
    "BTN_BG": "#ff7a00",
    "BTN_FG": "#1a1a1a",
    "BTN_ACTIVE_BG": "#ffc266",
    "BTN_ACTIVE_FG": "#1a1a1a",
}

# Festo signature palette (light mode).
LIGHT_PALETTE = {
    "BG_DARK": "#ffffff",
    "BG_PANEL": "#eef2f6",
    "ACCENT_ORANGE": "#0091c8",   # Festo blue as primary accent
    "ACCENT_YELLOW": "#f39200",   # Festo orange as secondary accent
    "CREAM": "#003366",           # Festo dark blue, used as "text" color
    "BTN_BG": "#0091c8",
    "BTN_FG": "#ffffff",
    "BTN_ACTIVE_BG": "#f39200",
    "BTN_ACTIVE_FG": "#ffffff",
}

CURRENT_THEME = "dark"

# Live module-level references - repointed by apply_theme().
BG_DARK = DARK_PALETTE["BG_DARK"]
BG_PANEL = DARK_PALETTE["BG_PANEL"]
ACCENT_ORANGE = DARK_PALETTE["ACCENT_ORANGE"]
ACCENT_YELLOW = DARK_PALETTE["ACCENT_YELLOW"]
CREAM = DARK_PALETTE["CREAM"]
BTN_BG = DARK_PALETTE["BTN_BG"]
BTN_FG = DARK_PALETTE["BTN_FG"]
BTN_ACTIVE_BG = DARK_PALETTE["BTN_ACTIVE_BG"]
BTN_ACTIVE_FG = DARK_PALETTE["BTN_ACTIVE_FG"]

_theme_listeners = []


def on_theme_change(callback):
    _theme_listeners.append(callback)


def apply_theme(theme_name):
    global CURRENT_THEME, BG_DARK, BG_PANEL, ACCENT_ORANGE, ACCENT_YELLOW, CREAM
    global BTN_BG, BTN_FG, BTN_ACTIVE_BG, BTN_ACTIVE_FG

    palette = LIGHT_PALETTE if theme_name == "light" else DARK_PALETTE
    CURRENT_THEME = theme_name

    BG_DARK = palette["BG_DARK"]
    BG_PANEL = palette["BG_PANEL"]
    ACCENT_ORANGE = palette["ACCENT_ORANGE"]
    ACCENT_YELLOW = palette["ACCENT_YELLOW"]
    CREAM = palette["CREAM"]
    BTN_BG = palette["BTN_BG"]
    BTN_FG = palette["BTN_FG"]
    BTN_ACTIVE_BG = palette["BTN_ACTIVE_BG"]
    BTN_ACTIVE_FG = palette["BTN_ACTIVE_FG"]

    for cb in list(_theme_listeners):
        try:
            cb(theme_name)
        except Exception:
            pass


def get_theme():
    return CURRENT_THEME


# ----------------------------------------------------------------------
# Widget styling helpers (same call signatures as before)
# ----------------------------------------------------------------------

def style_button(btn):
    btn.configure(
        bg=BTN_BG, fg=BTN_FG,
        activebackground=BTN_ACTIVE_BG, activeforeground=BTN_ACTIVE_FG,
        font=FONT_BTN, relief="flat", bd=0,
        padx=15, pady=8, cursor="hand2"
    )


def style_window(win):
    win.configure(bg=BG_DARK)


def refresh_widget_tree(widget):
    """Recursively re-applies current theme colors to an existing
    widget tree - call after a theme switch so already-open windows
    update without needing to be reopened."""
    try:
        cls = widget.winfo_class()
        if cls in ("Toplevel", "Tk", "Frame"):
            widget.configure(bg=BG_DARK)
        elif cls == "Label":
            widget.configure(bg=BG_DARK)
        elif cls == "Button":
            style_button(widget)
        elif cls == "Canvas":
            widget.configure(bg=BG_PANEL)
    except Exception:
        pass

    for child in widget.winfo_children():
        refresh_widget_tree(child)
