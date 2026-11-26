import os
import shutil
import threading
from ultralytics import YOLO
from core.dataset_utils import build_yolo_dataset, list_products, BASE_DIR
from core.model_utils import get_latest_model_path, get_next_model_path, MODELS_DIR
from core.project_config import training_settings

# Absolute path: with a relative "project", Ultralytics silently nests the
# run under runs/segment/..., and best.pt was then never found/copied.
RUNS_DIR = os.path.join(BASE_DIR, "runs")
PRETRAINED_SEG = os.path.join(MODELS_DIR, "yolov8n-seg.pt")

# Only one training run at a time, even if the Training menu is closed
# and reopened while a run is still going.
_TRAIN_LOCK = threading.Lock()


def train(epochs=None, imgsz=None, patience=None, batch=None,
          progress_callback=None, warm_start=None):
    if not _TRAIN_LOCK.acquire(blocking=False):
        msg = "A training run is already in progress."
        print(msg)
        if progress_callback:
            progress_callback(0, epochs or 1, None, None, None, error=msg)
        return
    try:
        return _train(epochs, imgsz, patience, batch, progress_callback, warm_start)
    finally:
        _TRAIN_LOCK.release()


def _train(epochs, imgsz, patience, batch, progress_callback, warm_start):
    """
    Builds a YOLO-format dataset from all recorded sessions and trains a
    YOLOv8n-seg model. Any argument left as None is taken from the
    "training" section of project.json.

    Initial weights, in order of preference:
      1. warm_start=True and a models/cpx_model_vN.pt exists -> continue
         from the newest one (works even if the class list changed;
         Ultralytics rebuilds the head for the new number of classes).
      2. models/yolov8n-seg.pt (COCO-pretrained, ships with the project).
      3. yolov8n-seg.yaml (random init) as a last resort.

    Saves the result to the NEXT version number (e.g. cpx_model_v8.pt),
    never overwriting or deleting older versions.

    progress_callback(epoch, total_epochs, box_loss, cls_loss, map50, error=None)
    """
    cfg = training_settings()
    epochs = epochs if epochs is not None else int(cfg["epochs"])
    imgsz = imgsz if imgsz is not None else int(cfg["imgsz"])
    patience = patience if patience is not None else int(cfg["patience"])
    batch = batch if batch is not None else int(cfg["batch"])
    warm_start = warm_start if warm_start is not None else bool(cfg["warm_start"])

    products = list_products()
    if len(products) < 1:
        msg = "No products recorded yet."
        print(msg)
        if progress_callback:
            progress_callback(0, epochs, None, None, None, error=msg)
        return

    try:
        data_yaml = build_yolo_dataset(val_split=0.2)
    except ValueError as e:
        data_yaml = None
        print(e)
    if data_yaml is None:
        msg = "No labeled training samples found. Record some sessions first."
        print(msg)
        if progress_callback:
            progress_callback(0, epochs, None, None, None, error=msg)
        return

    previous_model = get_latest_model_path() if warm_start else None

    if previous_model:
        print(f"Warm-starting training from previous model: {previous_model}")
        model = YOLO(previous_model)
    elif os.path.exists(PRETRAINED_SEG):
        print(f"Starting from COCO-pretrained weights: {PRETRAINED_SEG}")
        model = YOLO(PRETRAINED_SEG)
    else:
        print("No weights found - training from random init (yolov8n-seg.yaml).")
        model = YOLO("yolov8n-seg.yaml")

    if progress_callback:
        def on_epoch_end(trainer):
            try:
                metrics = trainer.metrics
                items = trainer.tloss if getattr(trainer, "tloss", None) is not None else trainer.loss_items
                if isinstance(items, dict):
                    # Ultralytics 8.4+: {"box_loss": t, "seg_loss": t, "cls_loss": t, ...}
                    box_loss = float(items.get("box_loss")) if "box_loss" in items else None
                    cls_loss = float(items.get("cls_loss")) if "cls_loss" in items else None
                elif items is not None:
                    # older versions: tensor [box, seg, cls, dfl] (seg) / [box, cls, dfl] (detect)
                    box_loss = float(items[0])
                    cls_loss = float(items[2] if len(items) > 3 else items[1])
                else:
                    box_loss = cls_loss = None
                map50 = metrics.get("metrics/mAP50(B)", None) if metrics else None
                progress_callback(trainer.epoch + 1, trainer.epochs, box_loss, cls_loss, map50)
            except Exception as e:
                print(f"Progress callback error: {e}")

        model.add_callback("on_fit_epoch_end", on_epoch_end)

    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
            workers=0,  # Windows + Tkinter thread: avoid dataloader subprocesses
            project=RUNS_DIR,
            name="cpx_detector",
            exist_ok=True,
            verbose=True,
        )
    except Exception as e:
        msg = f"Training failed: {e}"
        print(msg)
        if progress_callback:
            progress_callback(0, epochs, None, None, None, error=msg)
        return

    # Copy best weights to the next version's stable, known path.
    best_path = str(getattr(model.trainer, "best", "") or "")
    if not os.path.exists(best_path):
        best_path = os.path.join(RUNS_DIR, "cpx_detector", "weights", "best.pt")
    if os.path.exists(best_path):
        next_path = get_next_model_path()
        shutil.copy(best_path, next_path)
        print(f"Model saved to {next_path}")
    else:
        print("Warning: best.pt not found after training.")
