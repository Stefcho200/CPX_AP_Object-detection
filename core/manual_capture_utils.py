import os
import shutil
import datetime
import cv2
import numpy as np
from core.dataset_utils import DATA_DIR, new_session_path

TEMP_DIR = os.path.join(os.path.dirname(DATA_DIR), "manual_capture_temp")


def new_temp_batch(product_name):
    """Creates a temp (not-yet-committed) capture batch folder, mirroring
    a normal session's images/+labels/ structure. Nothing here ever
    touches the real dataset until commit_batch() is called."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_path = os.path.join(TEMP_DIR, product_name, f"batch_{ts}")
    suffix = 2
    while os.path.exists(batch_path):
        batch_path = os.path.join(TEMP_DIR, product_name, f"batch_{ts}_{suffix}")
        suffix += 1
    os.makedirs(os.path.join(batch_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(batch_path, "labels"), exist_ok=True)
    return batch_path


def list_pending_batches():
    """Scans the temp capture area for any batches that were never
    committed or discarded (e.g. left over from a crash) and returns
    [(product_name, batch_path, frame_count), ...]."""
    pending = []
    if not os.path.isdir(TEMP_DIR):
        return pending
    for product in sorted(os.listdir(TEMP_DIR)):
        product_dir = os.path.join(TEMP_DIR, product)
        if not os.path.isdir(product_dir):
            continue
        for batch_name in sorted(os.listdir(product_dir)):
            batch_path = os.path.join(product_dir, batch_name)
            images_dir = os.path.join(batch_path, "images")
            if os.path.isdir(images_dir):
                count = len(os.listdir(images_dir))
                if count > 0:
                    pending.append((product, batch_path, count))
    return pending



def list_batch_frames(batch_path):
    images_dir = os.path.join(batch_path, "images")
    labels_dir = os.path.join(batch_path, "labels")
    frames = []
    if not os.path.isdir(images_dir):
        return frames
    for fname in sorted(os.listdir(images_dir)):
        base = os.path.splitext(fname)[0]
        frames.append({
            "image_path": os.path.join(images_dir, fname),
            "label_path": os.path.join(labels_dir, base + ".txt"),
        })
    return frames


def auto_label_batch(batch_path, detector, conf_threshold=0.5, progress_callback=None):
    """Runs the given detector over every raw captured frame in the batch
    and writes a label file for any frame where a mask was found. Frames
    the model misses are simply left without a label file — the review
    window treats those as 'NO MASK, needs manual fix' automatically,
    no separate flagging step needed."""
    frames = list_batch_frames(batch_path)
    total = len(frames)
    labeled = 0
    detector.conf_threshold = conf_threshold

    for i, f in enumerate(frames):
        img = cv2.imread(f["image_path"])
        if img is not None:
            h_img, w_img = img.shape[:2]
            detections = detector.predict(img)
            if detections:
                _, conf, bbox, polygon = max(detections, key=lambda d: d[1])
                if polygon is not None and len(polygon) >= 3:
                    pts = np.array(polygon, dtype=np.float32)
                    pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)
                    norm = []
                    for px, py in pts:
                        norm += [px / w_img, py / h_img]
                    with open(f["label_path"], "w") as out:
                        out.write("0 " + " ".join(f"{v:.6f}" for v in norm) + "\n")
                    labeled += 1

        if progress_callback:
            progress_callback(i + 1, total)

    return labeled, total


def write_manual_label(frame_info, polygon, frame_shape):
    """Overwrites a single frame's label with a manually/SAM-produced
    polygon (pixel coords), used by the review window's manual-fix mode."""
    h_img, w_img = frame_shape[:2]
    pts = np.array(polygon, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)

    norm = []
    for px, py in pts:
        norm += [px / w_img, py / h_img]

    with open(frame_info["label_path"], "w") as f:
        f.write("0 " + " ".join(f"{v:.6f}" for v in norm) + "\n")


def remove_frame(frame_info):
    """Deletes a frame + its label entirely from the temp batch (used
    when a captured frame is just bad and not worth keeping at all)."""
    for key in ("image_path", "label_path"):
        p = frame_info.get(key)
        if p and os.path.exists(p):
            os.remove(p)


def commit_batch(batch_path, product_name):
    """Moves every frame that currently HAS a valid label into a brand
    new real session for this product, renumbering files sequentially.
    Frames still missing a label (never auto-labeled, never manually
    fixed) are left behind and discarded along with the temp batch —
    they never make it into the real dataset. Returns (session_path,
    committed_count)."""
    session_path = new_session_path(product_name)
    frames = list_batch_frames(batch_path)
    count = 0

    for f in frames:
        if not os.path.exists(f["label_path"]):
            continue
        fname = f"{count:05d}"
        shutil.move(f["image_path"], os.path.join(session_path, "images", fname + ".jpg"))
        shutil.move(f["label_path"], os.path.join(session_path, "labels", fname + ".txt"))
        count += 1

    shutil.rmtree(batch_path, ignore_errors=True)
    return session_path, count


def discard_batch(batch_path):
    """Throws away the entire temp batch — nothing was ever committed
    to the real dataset, so this is a full no-op on real data."""
    shutil.rmtree(batch_path, ignore_errors=True)


