import os
import shutil
from ultralytics import YOLO
from core.dataset_utils import build_yolo_dataset, list_products
from core.model_utils import get_latest_model_path, get_next_model_path

RUNS_DIR = "runs_cpx"


def train(epochs=1000, imgsz=640, patience=50, progress_callback=None, warm_start=True):

    """
    Builds a YOLO-format dataset from all recorded sessions and trains a
    YOLOv8 detector.

    warm_start=True (default): if a previous cpx_model_vN.pt exists on
    disk, initializes training from THOSE weights instead of random init.
    This is loading a LOCAL file already on disk (not downloading
    Ultralytics' hosted pretrained weights), so it works fine despite the
    corporate SSL/proxy restriction. Even though the previous model may
    have been trained on fewer products / dirtier data, its low-level
    learned features (edges, shine, rectangularity) should still give the
    new training run a useful head start over starting from scratch.

    Saves the result to the NEXT version number (e.g. cpx_model_v3.pt),
    never overwriting or deleting older versions.

    progress_callback(epoch, total_epochs, box_loss, cls_loss, map50, error=None)
    """
    products = list_products()
    if len(products) < 1:
        msg = "No products recorded yet."
        print(msg)
        if progress_callback:
            progress_callback(0, epochs, None, None, None, error=msg)
        return

    data_yaml = build_yolo_dataset(val_split=0.2)
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
    else:
        print("No previous model found — training from random init (yolov8n-seg.yaml).")
        model = YOLO("yolov8n-seg.yaml")


    if progress_callback:
        def on_epoch_end(trainer):
            try:
                metrics = trainer.metrics
                box_loss = float(trainer.loss_items[0]) if trainer.loss_items is not None else None
                cls_loss = float(trainer.loss_items[1]) if trainer.loss_items is not None else None
                map50 = metrics.get("metrics/mAP50(B)", None) if metrics else None
                progress_callback(trainer.epoch + 1, trainer.epochs, box_loss, cls_loss, map50)
            except Exception as e:
                print(f"Progress callback error: {e}")

        model.add_callback("on_train_epoch_end", on_epoch_end)

    model.train(
        data=data_yaml,
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        project=RUNS_DIR,
        name="cpx_detector",
        exist_ok=True,
        verbose=True,
    )


    # Copy best weights to the next version's stable, known path.
    best_path = os.path.join(RUNS_DIR, "cpx_detector", "weights", "best.pt")
    if os.path.exists(best_path):
        next_path = get_next_model_path()
        shutil.copy(best_path, next_path)
        print(f"Model saved to {next_path}")
    else:
        print("Warning: best.pt not found after training.")
