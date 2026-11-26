import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(BASE_DIR, "record_settings.json")

DEFAULTS = {
    "threshold": 127,
    "blur": 5,
    "invert": False,
    # 0 = save the mask exactly as detected (was 20, which inflated every
    # new product's masks by 20 px in AI mode).
    "margin": 0,
    "morph": 3,
    "min_area": 500,
    "max_area": 200000,
    "roi_margin": 0,
    "solidity": 0,
}


def _load_all():
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def get_settings(product):
    """Returns saved settings for this product, falling back to global
    'default' settings, falling back to hardcoded DEFAULTS if neither exist."""
    all_settings = _load_all()
    if product in all_settings:
        merged = DEFAULTS.copy()
        merged.update(all_settings[product])
        return merged
    if "default" in all_settings:
        merged = DEFAULTS.copy()
        merged.update(all_settings["default"])
        return merged
    return DEFAULTS.copy()


def has_saved_settings(product):
    return product in _load_all()


def save_settings(product, settings_dict):
    all_settings = _load_all()
    all_settings[product] = settings_dict
    with open(SETTINGS_PATH, "w") as f:
        json.dump(all_settings, f, indent=2)
