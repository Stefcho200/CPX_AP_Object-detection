import os
import json

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_settings.json")

DEFAULTS = {
    "language": "en",
    "theme": "dark",
}


def load_settings():
    if not os.path.exists(SETTINGS_PATH):
        return dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
        merged = dict(DEFAULTS)
        merged.update(data)
        return merged
    except Exception:
        return dict(DEFAULTS)


def save_settings(settings):
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def apply_saved_settings():
    """Call once at app startup - loads persisted language/theme and
    applies them before any window is built."""
    from core import i18n
    import theme

    settings = load_settings()
    i18n.set_language(settings.get("language", "en"))
    theme.apply_theme(settings.get("theme", "dark"))
    return settings


def update_setting(key, value):
    settings = load_settings()
    settings[key] = value
    save_settings(settings)
