import time
import threading
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from core.infer import CPXDetector
from core.camera_utils import open_camera
from core.twin_groups import group_for_label
from core import ocr_utils
from core.i18n import t, tf, on_language_change
import theme
from gui.manual_label_window import ZoomableCanvas


def iou(box_a, box_b):
    """Standard IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


class DetectWindow(tk.Toplevel):

       # --- Lock/track state machine parameters ---
    LOCK_CONF_THRESHOLD = 0.65
    TRACK_CONF_THRESHOLD = 0.15
    MIN_TRACK_IOU = 0.1
    LOSS_TIMEOUT = 0.5

    # --- NEW: distance/confirmation safeguards against locking too far away ---
    MIN_LOCK_BOX_FRACTION = 0.15   # bbox area must cover at least 15% of the frame to lock
    LOCK_CONFIRM_FRAMES = 3         # must see the SAME class clear threshold this many consecutive frames


    # --- OCR disambiguation tuning (twin products only) ---
    OCR_RECHECK_INTERVAL = 0.75
    OCR_CONF_MIN = 0.35

    def __init__(self, master):
        super().__init__(master)
        self.title(t("detect_window_title"))
        theme.style_window(self)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.cap = open_camera()
        self.detector = CPXDetector(conf_threshold=self.LOCK_CONF_THRESHOLD)

        # --- Lock/track state ---
        self.state = "SEARCHING"
        self.locked_label = None
        self.last_bbox = None
        self.last_polygon = None
        self.last_seen_time = None

        # --- NEW: pre-lock confirmation streak ---
        self.candidate_label = None
        self.candidate_streak = 0


        # --- OCR disambiguation state ---
        self.last_ocr_time = 0.0
        self.ocr_resolved_label = None

        video_frame = tk.Frame(self, bg=theme.BG_PANEL)
        video_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = ZoomableCanvas(video_frame, bg=theme.BG_PANEL)
        self.canvas.pack(fill="both", expand=True)

        text_frame = tk.Frame(self, bg=theme.BG_DARK, height=60)
        text_frame.pack(fill="x")
        text_frame.pack_propagate(False)

        self.result_label = tk.Label(text_frame, text="", bg=theme.BG_DARK,
                              fg=theme.ACCENT_ORANGE, font=theme.FONT_TITLE,
                              wraplength=600, justify="center")
        self.result_label.pack(expand=True)

        if not self.detector.loaded:
            self.result_label.configure(
                text=tf("no_model_feed", error=self.detector.load_error),
                fg=theme.ACCENT_YELLOW)
        else:
            self.result_label.configure(
                text=t("hold_product_prompt"), fg=theme.ACCENT_ORANGE)

        self.close_btn = tk.Button(self, text=t("close"), command=self.on_close)
        theme.style_button(self.close_btn)
        self.close_btn.pack(pady=10)

        self._unsub_lang = on_language_change(self._on_language_changed)

        # Warm up EasyOCR in the background so the first twin-product
        # encounter during live detection doesn't stutter on model load.
        threading.Thread(target=ocr_utils.warm_up, daemon=True).start()

        self.update_frame()

    # ------------------------------------------------------------------
    # Language handling
    # ------------------------------------------------------------------

    def _on_language_changed(self, lang_code):
        self.title(t("detect_window_title"))
        self.close_btn.configure(text=t("close"))
        # The result_label's text is refreshed continuously by
        # update_frame() every ~30ms anyway, so no need to force it
        # here - it'll pick up the new language on the very next tick.

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_mask(self, display, polygon, color=(0, 122, 255), alpha=0.35):
        pts = np.array(polygon, dtype=np.int32)
        if pts.shape[0] < 3:
            return

        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, dst=display)
        cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)

    # ------------------------------------------------------------------
    # Lock/track state machine
    # ------------------------------------------------------------------

    def _reset_lock(self):
        self.state = "SEARCHING"
        self.locked_label = None
        self.last_bbox = None
        self.last_polygon = None
        self.last_seen_time = None
        self.last_ocr_time = 0.0
        self.ocr_resolved_label = None
        self.candidate_label = None
        self.candidate_streak = 0


    def _maybe_run_ocr(self, frame):
        if self.ocr_resolved_label is not None:
            return

        group = group_for_label(self.locked_label)
        if group is None or self.last_polygon is None:
            return

        now = time.time()
        if now - self.last_ocr_time < self.OCR_RECHECK_INTERVAL:
            return
        self.last_ocr_time = now

        crop = ocr_utils.get_oriented_crop(frame, self.last_polygon)
        if crop is None:
            return

        text, conf = ocr_utils.best_rotation_ocr(crop)
        if conf < self.OCR_CONF_MIN:
            return

        resolved = ocr_utils.match_text_to_product(text, group)
        if resolved is not None:
            self.ocr_resolved_label = resolved

    def _process_detections(self, detections, frame):
        now = time.time()

        if self.state == "SEARCHING":
            if not detections:
                self.candidate_label = None
                self.candidate_streak = 0
                return None

            best = max(detections, key=lambda d: d[1])
            label, conf, bbox, polygon = best

            if conf < self.LOCK_CONF_THRESHOLD:
                self.candidate_label = None
                self.candidate_streak = 0
                return None

            # Distance gate: reject locks on products that don't yet
            # occupy enough of the frame - a confident guess from far
            # away is still not trustworthy, since resolution/detail is
            # exactly what's missing at distance.

            #frame_area = frame.shape[0] * frame.shape[1]
            #x1, y1, x2, y2 = bbox
            #box_area = max(0, x2 - x1) * max(0, y2 - y1)
            #box_fraction = box_area / frame_area if frame_area > 0 else 0

            #if box_fraction < self.MIN_LOCK_BOX_FRACTION:
                self.candidate_label = None
                self.candidate_streak = 0
                return None

            # Confirmation streak: require the SAME class to pass both
            # checks for several consecutive frames before committing,
            # so a single lucky/unlucky spike can't lock in a wrong ID.
            if label == self.candidate_label:
                self.candidate_streak += 1
            else:
                self.candidate_label = label
                self.candidate_streak = 1

            if self.candidate_streak < self.LOCK_CONFIRM_FRAMES:
                return None

            # Passed all three gates - commit the lock.
            self.state = "LOCKED"
            self.locked_label = label
            self.last_bbox = bbox
            self.last_polygon = polygon
            self.last_seen_time = now
            self.last_ocr_time = 0.0
            self.ocr_resolved_label = None
            self.candidate_label = None
            self.candidate_streak = 0
            return best


            return None

        if self.state == "LOCKED":
            match = None
            best_iou = 0.0

            for det in detections:
                _, _, bbox, _ = det
                score = iou(bbox, self.last_bbox)
                if score > best_iou:
                    best_iou = score
                    match = det

            if match is not None and best_iou >= self.MIN_TRACK_IOU:
                _, conf, bbox, polygon = match
                self.last_bbox = bbox
                self.last_polygon = polygon
                self.last_seen_time = now

                self._maybe_run_ocr(frame)

                display_label = self.ocr_resolved_label or self.locked_label
                return (display_label, conf, bbox, polygon)

            if now - self.last_seen_time > self.LOSS_TIMEOUT:
                self._reset_lock()
                return None

            return None

        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def update_frame(self):
        if hasattr(self.cap, "grab"):
            # Standard cv2.VideoCapture (local webcam) - grab a couple of
            # frames to skip any small buffered backlog, then retrieve.
            for _ in range(2):
                self.cap.grab()
            ret, frame = self.cap.retrieve()
        else:
            # LatestFrameReader (IP camera) - already guarantees the
            # freshest frame with no backlog, just read it directly.
            ret, frame = self.cap.read()

        if ret:
            display = frame.copy()

            if self.detector.loaded:
                self.detector.conf_threshold = (
                    self.LOCK_CONF_THRESHOLD if self.state == "SEARCHING"
                    else self.TRACK_CONF_THRESHOLD
                )

                detections = self.detector.predict(frame)
                result = self._process_detections(detections, frame)

                if result is not None:
                    label, conf, (x1, y1, x2, y2), polygon = result

                    if polygon is not None:
                        self._draw_mask(display, polygon)
                    else:
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 122, 255), 2)

                    label_text = f"{label} {conf*100:.0f}%"
                    text_x, text_y = x1, max(0, y1 - 10)
                    cv2.putText(display, label_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 122, 255), 2)

                    status = t("tracking_status") if self.state == "LOCKED" else t("identified_status")
                    suffix = t("ocr_confirmed_suffix") if self.ocr_resolved_label else ""
                    self.result_label.configure(
                        text=f"{status}: {label} {conf*100:.0f}%{suffix}",
                        fg=theme.ACCENT_ORANGE)

                elif self.state == "LOCKED":
                    display_label = self.ocr_resolved_label or self.locked_label
                    self.result_label.configure(
                        text=tf("tracking_lost", label=display_label),
                        fg=theme.ACCENT_YELLOW)

                else:
                    self.result_label.configure(
                        text=t("hold_product_prompt"),
                        fg=theme.ACCENT_YELLOW)

            self.canvas.show(display)

        self.after(30, self.update_frame)




    def on_close(self):
        self.cap.release()
        self.destroy()
