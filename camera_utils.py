import cv2
import threading
import time
import numpy as np
from core.config_utils import (
    get_camera_index, get_camera_source, get_known_ips
)

MAX_INDEX_TO_SCAN = 4
BLACK_THRESHOLD = 5

# MOBOTIX cameras typically respond to one of these without needing a browser.
# We try MJPEG first (near-universal on MOBOTIX, no extra setup), then RTSP.
IP_STREAM_TEMPLATES = [
    "http://{ip}/control/faststream.jpg?stream=full&preview=1&size=640x480",
    "rtsp://{ip}/control",
]




class LatestFrameReader:
    """
    Runs a background thread that continuously reads frames from a
    cv2.VideoCapture and keeps only the most recent one. This prevents
    network streams (especially high-res MJPEG/RTSP) from building up
    an internal backlog that causes multi-second lag by the time the
    GUI thread gets around to reading a frame.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                # Small sleep to avoid a tight spin loop if the stream
                # hiccups momentarily.
                time.sleep(0.05)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()




def _is_frame_usable(frame):
    if frame is None:
        return False
    mean_val = np.mean(frame)
    return mean_val > BLACK_THRESHOLD


def try_open_ip_camera(ip, read_attempts=5):
    """
    Tries each known MOBOTIX stream URL pattern for the given IP.
    Returns (cap, preview_frame, working_url) on success,
    or (None, None, None) if nothing works.
    Caller is responsible for releasing `cap` when done with it.
    """
    for template in IP_STREAM_TEMPLATES:
        url = template.format(ip=ip)
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(read_attempts):
            ret, frame = cap.read()
            if ret and _is_frame_usable(frame):
                return cap, frame, url

        cap.release()

    return None, None, None


def probe_ip_camera(ip):
    """
    Quick usability check for an IP camera, used by the settings grid.
    Returns dict: {"ip": ip, "usable": bool, "preview": frame_or_None, "url": url_or_None}
    """
    cap, frame, url = try_open_ip_camera(ip, read_attempts=5)
    result = {
        "ip": ip,
        "usable": cap is not None,
        "preview": frame,
        "url": url,
    }
    if cap is not None:
        cap.release()
    return result


def list_available_cameras(max_index=MAX_INDEX_TO_SCAN):
    """
    Scans local camera indices AND previously known IP cameras.
    Returns a unified list of dicts, each tagged with "type":
      Local:  {"type": "local", "index": 0, "usable": True, "preview": frame}
      IP:     {"type": "ip", "ip": "192.168.1.50", "usable": True, "preview": frame, "url": "..."}
    """
    results = []

    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            results.append({"type": "local", "index": idx, "usable": False, "preview": None})
            continue

        preview = None
        usable = False
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                preview = frame
                if _is_frame_usable(frame):
                    usable = True
                    break

        cap.release()
        results.append({"type": "local", "index": idx, "usable": usable, "preview": preview})

    for ip in get_known_ips():
        ip_result = probe_ip_camera(ip)
        ip_result["type"] = "ip"
        results.append(ip_result)

    return results


def find_working_camera():
    """
    Priority order:
    1. User-selected source from config (local index OR ip - if set and usable).
    2. Auto-scan local indices for first non-black feed.
    3. Fallback to local index 0 even if black, to avoid total failure.
    Returns a dict describing the chosen source, e.g.:
      {"type": "local", "value": 0} or {"type": "ip", "value": "192.168.1.50"}
    """
    source = get_camera_source()

    if source["type"] == "ip" and source["value"]:
        ip = source["value"]
        cap, frame, url = try_open_ip_camera(ip)
        if cap is not None:
            cap.release()
            print(f"Using preferred IP camera: {ip}")
            return {"type": "ip", "value": ip}
        print(f"Preferred IP camera {ip} failed usability check, falling back to auto-scan")

    elif source["type"] == "index" and source["value"] is not None:
        preferred = source["value"]
        cap = cv2.VideoCapture(preferred, cv2.CAP_DSHOW)
        if cap.isOpened():
            for _ in range(5):
                ret, frame = cap.read()
                if ret and _is_frame_usable(frame):
                    cap.release()
                    print(f"Using preferred camera index: {preferred}")
                    return {"type": "local", "value": preferred}
        cap.release()
        print(f"Preferred camera {preferred} failed usability check, falling back to auto-scan")

    for idx in range(MAX_INDEX_TO_SCAN):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        usable = False
        for _ in range(3):
            ret, frame = cap.read()
            if ret and _is_frame_usable(frame):
                usable = True
                break

        cap.release()
        if usable:
            print(f"Auto-detected usable camera index: {idx}")
            return {"type": "local", "value": idx}

    print("No usable camera found, defaulting to local index 0")
    return {"type": "local", "value": 0}


def open_camera():
    """Returns an object with .read()/.release() pointed at the best available camera."""
    chosen = find_working_camera()

    if chosen["type"] == "ip":
        cap, frame, url = try_open_ip_camera(chosen["value"])
        if cap is not None:
            print(f"Opening IP camera: {chosen['value']} via {url}")
            return LatestFrameReader(cap)
        print(f"IP camera {chosen['value']} unavailable at open time, falling back to local index 0")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    idx = chosen["value"]
    print(f"Opening camera index: {idx}")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
