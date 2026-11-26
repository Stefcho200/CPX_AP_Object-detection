import os
import re
import glob

# Anchor everything to the project root (parent of core/), same fix as
# config_utils.py — never depend on the current working directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

VERSION_PATTERN = re.compile(r"cpx_model_v(\d+)\.pt$", re.IGNORECASE)


def _all_versions():
    """Returns sorted list of (version_number, full_path) for every
    cpx_model_vN.pt found in the models folder."""
    found = []
    for path in glob.glob(os.path.join(MODELS_DIR, "cpx_model_v*.pt")):
        m = VERSION_PATTERN.search(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found, key=lambda t: t[0])


def get_latest_model_path():
    """
    Returns the path to the highest-numbered existing model, or None if
    no model has ever been trained yet. This is what DetectWindow /
    CPXDetector and the "warm start" training logic should always call —
    never hardcode a filename anywhere else.
    """
    versions = _all_versions()
    if not versions:
        return None
    return versions[-1][1]


def get_next_model_path():
    """
    Returns the path the NEXT trained model should be saved to
    (e.g. cpx_model_v3.pt if v1 and v2 already exist). Never overwrites
    or deletes old versions.
    """
    versions = _all_versions()
    next_version = (versions[-1][0] + 1) if versions else 1
    return os.path.join(MODELS_DIR, f"cpx_model_v{next_version}.pt")


def list_all_models():
    """Returns list of (version_number, path) for every model on disk,
    oldest to newest — useful for a future 'model history' UI."""
    return _all_versions()
