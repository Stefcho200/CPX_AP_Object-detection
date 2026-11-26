import os
import json
import cv2
import numpy as np
from core.dataset_utils import DATA_DIR, list_products, list_sessions
import shutil


FLAGS_PATH = os.path.join(os.path.dirname(DATA_DIR), "cleanup_flags.json")
STAGING_DIR = os.path.join(os.path.dirname(DATA_DIR), "relabel_staging")



# ----------------------------------------------------------------------
# Frame listing across products/sessions, without ever duplicating data
# ----------------------------------------------------------------------

def list_frames(scope="all", product=None, session=None):
    """Returns a flat, ordered list of frame dicts:
        {"product": ..., "session": ..., "image_path": ..., "label_path": ...}

    scope="all"     -> every product, every session, in order.
    scope="product" -> every session of `product`, in order.
    scope="session" -> just `session` under `product`.

    This is the shared backbone for both the cleanup viewer (playback)
    and the re-labeling tool (batch processing) — neither ever copies
    images/videos, they just walk this list and touch files in place.
    """
    frames = []

    if scope == "session":
        products_to_scan = [(product, [session])]
    elif scope == "product":
        products_to_scan = [(product, list_sessions(product))]
    else:
        products_to_scan = [(p, list_sessions(p)) for p in list_products()]

    for prod, sessions in products_to_scan:
        for sess in sessions:
            images_dir = os.path.join(DATA_DIR, prod, sess, "images")
            labels_dir = os.path.join(DATA_DIR, prod, sess, "labels")
            if not os.path.isdir(images_dir):
                continue
            for fname in sorted(os.listdir(images_dir)):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(images_dir, fname)
                lbl_path = os.path.join(labels_dir, base + ".txt")
                frames.append({
                    "product": prod,
                    "session": sess,
                    "image_path": img_path,
                    "label_path": lbl_path,
                })

    return frames


# ----------------------------------------------------------------------
# Flags: persisted separately from the actual dataset files, so flagging
# a frame for review never touches/renames the real image or label.
# ----------------------------------------------------------------------

def load_flags():
    if not os.path.exists(FLAGS_PATH):
        return set()
    try:
        with open(FLAGS_PATH, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_flags(flag_set):
    with open(FLAGS_PATH, "w") as f:
        json.dump(sorted(flag_set), f, indent=2)


def toggle_flag(image_path, flag_set):
    """Mutates flag_set in place and persists it. Returns new flagged state."""
    if image_path in flag_set:
        flag_set.discard(image_path)
        state = False
    else:
        flag_set.add(image_path)
        state = True
    save_flags(flag_set)
    return state


def delete_flagged(flag_set):
    """Deletes every flagged image + its matching label file, then clears
    the flag list. Returns count deleted."""
    deleted = 0
    for img_path in list(flag_set):
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted += 1
        base = os.path.splitext(img_path)[0]
        # image was at .../images/name.jpg -> label at .../labels/name.txt
        lbl_path = base.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}") + ".txt"
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
    flag_set.clear()
    save_flags(flag_set)
    return deleted


# ----------------------------------------------------------------------
# Re-labeling existing frames using the current best model
# ----------------------------------------------------------------------

def relabel_frames(frames, detector, conf_threshold=0.5, progress_callback=None,
                    auto_flag_missing=True, flag_set=None):
    """Runs the given CPXDetector over every frame in `frames` and writes
    the freshly predicted polygon to a STAGING copy of the label file,
    never touching the real dataset label. The original label in
    data/<product>/<session>/labels/ is left completely untouched until
    the user explicitly reviews and commits the staged results.

    Staged labels are stored at:
        relabel_staging/<product>/<session>/<frame>.txt
    mirroring the real dataset's structure, so commit/discard can walk
    both trees in lockstep.

    Returns (relabeled_count, flagged_count, staging_manifest) where
    staging_manifest is a list of dicts describing every frame that got
    a new staged label, for the review UI to consume directly.
    """
    detector.conf_threshold = conf_threshold
    relabeled = 0
    flagged = 0
    total = len(frames)
    manifest = []

    for i, frame_info in enumerate(frames):
        img_path = frame_info["image_path"]
        real_lbl_path = frame_info["label_path"]
        product = frame_info["product"]
        session = frame_info["session"]

        img = cv2.imread(img_path)
        if img is None:
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        h_img, w_img = img.shape[:2]
        detections = detector.predict(img)

        staged_lbl_path = os.path.join(
            STAGING_DIR, product, session,
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )

        if not detections:
            if auto_flag_missing and flag_set is not None:
                flag_set.add(img_path)
                flagged += 1
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        _, conf, bbox, polygon = max(detections, key=lambda d: d[1])

        if polygon is None or len(polygon) < 3:
            if auto_flag_missing and flag_set is not None:
                flag_set.add(img_path)
                flagged += 1
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        pts = np.array(polygon, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)

        norm_coords = []
        for px, py in pts:
            norm_coords.append(px / w_img)
            norm_coords.append(py / h_img)

        os.makedirs(os.path.dirname(staged_lbl_path), exist_ok=True)
        with open(staged_lbl_path, "w") as f:
            coords_str = " ".join(f"{v:.6f}" for v in norm_coords)
            f.write(f"0 {coords_str}\n")

        relabeled += 1
        manifest.append({
            "product": product,
            "session": session,
            "image_path": img_path,
            "real_label_path": real_lbl_path,
            "staged_label_path": staged_lbl_path,
        })

        if progress_callback:
            progress_callback(i + 1, total)

    if flag_set is not None:
        save_flags(flag_set)

    save_staging_manifest(manifest)
    return relabeled, flagged, manifest


# ----------------------------------------------------------------------
# Staging manifest persistence — lets the review window be reopened
# later without re-running the model, and lets commit/discard know
# exactly what belongs to the current staged batch.
# ----------------------------------------------------------------------

MANIFEST_PATH = os.path.join(STAGING_DIR, "_manifest.json")


def save_staging_manifest(manifest):
    os.makedirs(STAGING_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def load_staging_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return []
    try:
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def has_pending_staging():
    return len(load_staging_manifest()) > 0


def commit_staged_relabels(manifest=None):
    """Copies every staged label over its corresponding real label,
    making the re-labeling permanent. Then clears the staging area.
    Returns count committed."""
    if manifest is None:
        manifest = load_staging_manifest()

    committed = 0
    for entry in manifest:
        staged = entry["staged_label_path"]
        real = entry["real_label_path"]
        if os.path.exists(staged):
            os.makedirs(os.path.dirname(real), exist_ok=True)
            with open(staged, "r") as f_in, open(real, "w") as f_out:
                f_out.write(f_in.read())
            committed += 1

    clear_staging()
    return committed


def discard_staged_relabels():
    """Throws away all staged labels without touching the real dataset
    at all. Real labels remain exactly as they were before re-labeling
    was ever run."""
    clear_staging()


def clear_staging():
    if os.path.isdir(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)
