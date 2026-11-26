import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    # camera_source replaces the old flat camera_index concept.
    # type: "index" (local webcam/built-in) or "ip" (MOBOTIX/network camera)
    # value: int index, or ip string (e.g. "192.168.1.50"), or None for auto-detect
    "camera_source": {"type": "index", "value": None},
    # Remembers IP addresses that have successfully connected before,
    # so the user doesn't have to retype them.
    "known_ip_cameras": []
}


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        return merged
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------
# Camera source (unified: local index OR IP camera)
# ---------------------------------------------------------------------

def get_camera_source():
    """Returns dict: {"type": "index"/"ip", "value": ...}"""
    config = load_config()
    return config.get("camera_source", DEFAULT_CONFIG["camera_source"])


def set_camera_source(source_type, value):
    config = load_config()
    config["camera_source"] = {"type": source_type, "value": value}
    save_config(config)


# ---------------------------------------------------------------------
# Backward-compatible helpers (existing code calling these keeps working)
# ---------------------------------------------------------------------

def get_camera_index():
    source = get_camera_source()
    if source["type"] == "index":
        return source["value"]
    return None


def set_camera_index(index):
    set_camera_source("index", index)


# ---------------------------------------------------------------------
# Known IP camera history
# ---------------------------------------------------------------------

def get_known_ips():
    return load_config().get("known_ip_cameras", [])


def add_known_ip(ip):
    config = load_config()
    known = config.get("known_ip_cameras", [])
    if ip not in known:
        known.append(ip)
        config["known_ip_cameras"] = known
        save_config(config)


def remove_known_ip(ip):
    config = load_config()
    known = config.get("known_ip_cameras", [])
    if ip in known:
        known.remove(ip)
        config["known_ip_cameras"] = known
        save_config(config)
