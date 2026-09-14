import os
import shutil
import random
import cv2
import numpy as np

# Anchor DATA_DIR to this file's own location rather than the process's
# current working directory. Without this, "data" is resolved relative
# to wherever the terminal happened to `cd` to before launching the app —
# if that's the wrong folder (e.g. the parent "Python projects" dir
# instead of "cpx_recognizer"), list_products()/list_sessions() will
# silently look in (or create!) the wrong "data" folder, showing stale,
# empty, or partial results with no error at all.
#
# core/dataset_utils.py -> go up one level to reach the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")



# ----------------------------------------------------------------------
# Product / session management
# ----------------------------------------------------------------------

def create_product(product_name):
    """Creates the base folder for a new product, if it doesn't exist."""
    product_path = os.path.join(DATA_DIR, product_name)
    os.makedirs(product_path, exist_ok=True)
    return product_path


def list_products():
    """Returns a sorted list of product names found under DATA_DIR."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])


def list_sessions(product_name):
    """Returns a sorted list of session folder names for a given product."""
    product_path = os.path.join(DATA_DIR, product_name)
    if not os.path.isdir(product_path):
        return []
    return sorted([
        d for d in os.listdir(product_path)
        if os.path.isdir(os.path.join(product_path, d))
    ])

def count_images_per_product():
    """Returns {product_name: total_image_count} across all sessions,
    for the distribution chart / 'record more data' suggestion feature."""
    counts = {}
    for p in list_products():
        total = 0
        for s in list_sessions(p):
            img_dir = os.path.join(DATA_DIR, p, s, "images")
            if os.path.isdir(img_dir):
                total += len(os.listdir(img_dir))
        counts[p] = total
    return counts


def new_session_path(product_name):
    """Creates a new timestamp-based session folder (images/ + labels/)
    under the given product and returns its path."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = os.path.join(DATA_DIR, product_name, f"session_{timestamp}")
    os.makedirs(os.path.join(session_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "labels"), exist_ok=True)
    return session_path


def retag_session(session_path, new_product_name):
    """Moves an entire session folder from its current product to a
    different product folder (e.g. correcting a mislabeled recording)."""
    session_name = os.path.basename(session_path)
    new_product_path = os.path.join(DATA_DIR, new_product_name)
    os.makedirs(new_product_path, exist_ok=True)
    new_path = os.path.join(new_product_path, session_name)
    shutil.move(session_path, new_path)
    return new_path


def delete_session(session_path):
    """Permanently deletes a single session folder."""
    if os.path.isdir(session_path):
        shutil.rmtree(session_path)


def delete_product(product_name):
    """Permanently deletes an entire product folder (all its sessions)."""
    product_path = os.path.join(DATA_DIR, product_name)
    if os.path.isdir(product_path):
        shutil.rmtree(product_path)


# ----------------------------------------------------------------------
# Frame pruning: reduce redundant/near-duplicate training frames
# ----------------------------------------------------------------------

def _frame_hash(img_path, hash_size=16):
    """Computes a cheap perceptual-ish hash: downscale to a tiny grayscale
    thumbnail and use its raw pixel values. Frames that are visually
    near-identical (e.g. product held still for half a second) will
    produce near-identical hashes, while genuinely different poses/
    lighting will not."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    small = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.flatten()


def _hash_distance(h1, h2):
    """Mean absolute pixel difference between two small thumbnails — a
    simple, fast stand-in for perceptual similarity between two frames."""
    if h1 is None or h2 is None:
        return float("inf")
    return float(np.mean(np.abs(h1.astype(int) - h2.astype(int))))


def _remove_pair(images_dir, labels_dir, image_filename):
    """Deletes an image and its matching label file, if present."""
    img_path = os.path.join(images_dir, image_filename)
    base = os.path.splitext(image_filename)[0]
    lbl_path = os.path.join(labels_dir, base + ".txt")
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists(lbl_path):
        os.remove(lbl_path)


def prune_session(session_path, mode="dedupe", decimate_n=3, similarity_threshold=3.0):
    """Reduces redundancy in a recorded session.

    mode="decimate": keeps only every Nth frame (simple, fast, but blind
        to content — may keep two near-identical frames or discard one
        that was genuinely different, purely based on position in the
        sequence).

    mode="dedupe": walks frames in chronological order and removes any
        frame that is near-identical (mean pixel difference on a small
        thumbnail below similarity_threshold) to the last KEPT frame.
        This adapts to actual content — if the product was moved/rotated
        quickly, few frames get removed; if it was held still, most of
        that stretch does, since those frames are actually redundant.

    Returns (kept_count, removed_count).
    """
    images_dir = os.path.join(session_path, "images")
    labels_dir = os.path.join(session_path, "labels")

    if not os.path.isdir(images_dir):
        return 0, 0

    filenames = sorted(os.listdir(images_dir))
    removed = 0
    kept = 0

    if mode == "decimate":
        for i, fname in enumerate(filenames):
            if i % decimate_n != 0:
                _remove_pair(images_dir, labels_dir, fname)
                removed += 1
            else:
                kept += 1

    elif mode == "dedupe":
        last_kept_hash = None
        for fname in filenames:
            img_path = os.path.join(images_dir, fname)
            h = _frame_hash(img_path)
            if last_kept_hash is not None and _hash_distance(h, last_kept_hash) < similarity_threshold:
                _remove_pair(images_dir, labels_dir, fname)
                removed += 1
            else:
                last_kept_hash = h
                kept += 1
    else:
        raise ValueError(f"Unknown prune mode: {mode}")

    return kept, removed


def prune_product(product_name, mode="dedupe", decimate_n=3, similarity_threshold=3.0):
    """Convenience wrapper: runs prune_session across every session
    belonging to a product. Returns (total_kept, total_removed)."""
    total_kept = 0
    total_removed = 0
    for session in list_sessions(product_name):
        session_path = os.path.join(DATA_DIR, product_name, session)
        kept, removed = prune_session(
            session_path, mode=mode, decimate_n=decimate_n,
            similarity_threshold=similarity_threshold
        )
        total_kept += kept
        total_removed += removed
    return total_kept, total_removed


# ----------------------------------------------------------------------
# YOLO dataset building
# ----------------------------------------------------------------------

def build_yolo_dataset(output_dir="yolo_dataset", val_split=0.2, seed=42):
    """Walks all products/sessions under DATA_DIR, collapses twin-group
    products into shared meta-classes (see core/twin_groups.py), assigns
    a numeric class id per resulting class (sorted alphabetically for
    determinism), and writes a YOLO-format dataset (images/train,
    images/val, labels/train, labels/val, plus a data.yaml) into
    output_dir.

    Twin products (visually near-identical, e.g. the M8-3P 4-way group)
    are merged into ONE training class each, since asking the model to
    tell them apart from pixels alone is both impossible and actively
    harmful to training - it just gives the model contradictory signal
    on identical-looking inputs. OCR-based disambiguation (core/ocr_utils.py)
    resolves the specific twin at inference time instead; see
    twin_groups.group_for_label() which recognizes both real product
    names and merged meta-class names.

    Split is done at the frame level (random shuffle across all frames of
    a product), not at the session level — see the caveat discussed
    elsewhere about this potentially inflating validation metrics if
    frames from the same session end up in both train and val.
    """
    from core.twin_groups import build_class_remap

    products = list_products()
    if not products:
        raise ValueError("No products found to build a dataset from.")

    remap = build_class_remap(products)
    class_names = sorted(set(remap.values()))
    class_map = {name: idx for idx, name in enumerate(class_names)}

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    rng = random.Random(seed)

    for product in products:
        meta_class = remap[product]
        class_id = class_map[meta_class]
        sessions = list_sessions(product)

        all_frames = []
        for session in sessions:
            images_dir = os.path.join(DATA_DIR, product, session, "images")
            labels_dir = os.path.join(DATA_DIR, product, session, "labels")
            if not os.path.isdir(images_dir):
                continue
            for fname in os.listdir(images_dir):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(images_dir, fname)
                lbl_path = os.path.join(labels_dir, base + ".txt")
                if os.path.exists(lbl_path):
                    all_frames.append((img_path, lbl_path))

        rng.shuffle(all_frames)
        split_idx = int(len(all_frames) * (1 - val_split))
        train_frames = all_frames[:split_idx]
        val_frames = all_frames[split_idx:]

        for split_name, frames in (("train", train_frames), ("val", val_frames)):
            for img_path, lbl_path in frames:
                base = os.path.splitext(os.path.basename(img_path))[0]
                unique_name = f"{product}_{base}"

                dst_img = os.path.join(output_dir, "images", split_name, unique_name + ".jpg")
                dst_lbl = os.path.join(output_dir, "labels", split_name, unique_name + ".txt")

                shutil.copy(img_path, dst_img)

                with open(lbl_path, "r") as f_in, open(dst_lbl, "w") as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) < 7 or len(parts) % 2 == 0:
                            continue
                        coords = parts[1:]
                        f_out.write(f"{class_id} {' '.join(coords)}\n")

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    return yaml_path
