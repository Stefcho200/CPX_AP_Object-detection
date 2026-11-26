import os
import time
import multiprocessing as mp
from core.sam_worker_process import sam_worker_main

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOCAL_SAM_PATH = os.path.join(MODELS_DIR, "sam_b.pt")
LOCAL_MOBILE_SAM_PATH = os.path.join(MODELS_DIR, "mobile_sam.pt")

STARTUP_TIMEOUT = 60      # seconds to wait for the subprocess to load SAM
CALL_TIMEOUT = 30         # seconds to wait for a single prediction


class SAMHelper:
    """Manages a SAM subprocess and talks to it over queues. If the
    subprocess dies (crash, OOM kill, etc.) or a call times out, this
    transparently restarts it — the caller just gets None back for that
    one call and a clear error, instead of the whole app going down.

    This is a drop-in replacement for the old in-process SAMHelper: same
    predict_from_points() signature, same behavior from the caller's
    point of view, just crash-proof now."""

    def __init__(self, prefer_mobile=True):
        self._prefer_mobile = prefer_mobile
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._request_id = 0

        self.loaded = False
        self.error = None
        self.model_name = None

        self._resolve_weights()
        self._start_worker()

    def _resolve_weights(self):
        candidates = []
        if self._prefer_mobile:
            candidates.append(LOCAL_MOBILE_SAM_PATH)
        candidates.append(LOCAL_SAM_PATH)

        for local_path in candidates:
            if os.path.exists(local_path):
                self._weights_path = local_path
                self.model_name = os.path.basename(local_path)
                return

        self._weights_path = "mobile_sam.pt" if self._prefer_mobile else "sam_b.pt"
        self.model_name = self._weights_path

    def _start_worker(self):
        self._request_queue = mp.Queue()
        self._response_queue = mp.Queue()
        self._process = mp.Process(
            target=sam_worker_main,
            args=(self._request_queue, self._response_queue, self._weights_path),
            daemon=True,
        )
        self._process.start()

        try:
            kind, payload = self._response_queue.get(timeout=STARTUP_TIMEOUT)
        except Exception:
            self.loaded = False
            self.error = "SAM worker process did not start in time."
            self._kill_process()
            return

        if kind == "ready":
            self.loaded = True
            self.error = None
        else:
            self.loaded = False
            self.error = (
                f"{payload}\n\nIf this is a download/SSL error, manually download "
                f"mobile_sam.pt (recommended, much faster) or sam_b.pt from a "
                f"machine with internet access and place it at:\n"
                f"{LOCAL_MOBILE_SAM_PATH}\nor\n{LOCAL_SAM_PATH}"
            )
            self._kill_process()

    def _kill_process(self):
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._process = None

    def _is_alive(self):
        return self._process is not None and self._process.is_alive()

    def _restart(self):
        self._kill_process()
        self._start_worker()

    def predict_from_points(self, frame, points_xy, labels, frame_id=None):
        """Returns (polygon_or_None, error_message_or_None). error is
        None on success. If the worker died or timed out, this restarts
        it automatically and returns an error for THIS call only —
        the caller should just let the user click again."""
        if not self.loaded or not self._is_alive():
            self._restart()
            if not self.loaded:
                return None, self.error or "SAM worker is not available."

        self._request_id += 1
        req_id = self._request_id

        try:
            self._request_queue.put((req_id, frame, points_xy, labels))
        except Exception as e:
            self._restart()
            return None, f"Failed to send request to SAM worker: {e}"

        start = time.time()
        while time.time() - start < CALL_TIMEOUT:
            if not self._is_alive():
                self._restart()
                return None, "SAM worker process crashed during this prediction. It has been restarted — please try clicking again."

            try:
                kind, payload = self._response_queue.get(timeout=0.5)
            except Exception:
                continue  # nothing yet, keep waiting until CALL_TIMEOUT

            if kind == "result":
                resp_id, polygon = payload
                if resp_id != req_id:
                    continue  # stale response from a previous call, ignore and keep waiting
                return polygon, None

            if kind == "call_error":
                resp_id, err = payload
                self._restart()
                return None, f"SAM error: {err}\n\nWorker has been restarted."

        # Timed out waiting — assume the worker is stuck, restart it.
        self._restart()
        return None, "SAM took too long to respond and was restarted. Please try again."

    def shutdown(self):
        """Call when the app/window is closing, to cleanly stop the
        subprocess instead of leaving an orphaned python.exe running."""
        if self._request_queue is not None:
            try:
                self._request_queue.put(None)
            except Exception:
                pass
        self._kill_process()
