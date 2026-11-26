import os
from ultralytics import YOLO
from core.model_utils import get_latest_model_path


class CPXDetector:
    def __init__(self, conf_threshold=0.5):
        self.loaded = False
        self.load_error = None
        self.conf_threshold = conf_threshold
        self.model_path = get_latest_model_path()

        if self.model_path is not None and os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                self.loaded = True
                print(f"Loaded model: {self.model_path}")
            except Exception as e:
                self.model = None
                self.loaded = False
                self.load_error = str(e)
                print(f"Failed to load model: {e}")
        else:
            self.model = None
            self.load_error = "No trained model found yet."

    def class_names(self):
        if not self.loaded:
            return []
        return list(self.model.names.values())

    def knows_label(self, product_name):
        """True if the loaded model was trained on this product (directly,
        or through the twin meta-class the product belongs to)."""
        names = set(self.class_names())
        if product_name in names:
            return True
        from core.twin_groups import group_for_label, meta_class_name
        group = group_for_label(product_name)
        return group is not None and meta_class_name(group) in names

    def predict(self, frame):
        """
        Runs detection on a single BGR frame (numpy array).
        Returns a list of detections:
            [(label, confidence, (x1, y1, x2, y2), mask_polygon), ...]
        mask_polygon is an (N, 2) numpy array of pixel coords in the
        ORIGINAL frame's resolution, or None if the loaded model is a
        plain detector (no segmentation head) or has no mask for that box.
        Empty list if nothing detected above threshold, or model not loaded.
        """
        if not self.loaded:
            return []

        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        detections = []

        if len(results) > 0:
            r = results[0]
            names = r.names

            # r.masks.xy gives per-instance polygons already scaled back
            # to the original frame's resolution, in the same order as
            # r.boxes - so we can zip them together directly.
            polygons = r.masks.xy if r.masks is not None else None

            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = names[cls_id]

                polygon = None
                if polygons is not None and i < len(polygons):
                    polygon = polygons[i]  # numpy array of (x, y) points

                detections.append(
                    (label, conf, (int(x1), int(y1), int(x2), int(y2)), polygon)
                )

        return detections
