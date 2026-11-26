"""OCR-based disambiguation for visually-identical 'twin' CPX products.
Only invoked when the locked class is part of a twin group (see
core/twin_groups.py) - reads the small printed part code on the
product's label to resolve which twin it actually is.
"""

import re
import cv2
import numpy as np

_reader = None
_reader_error = None


def _get_reader():
    """Lazily initializes EasyOCR (slow to load, so only do it once,
    and only if/when actually needed)."""
    global _reader, _reader_error
    if _reader is not None or _reader_error is not None:
        return _reader
    try:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    except Exception as e:
        _reader_error = str(e)
        _reader = None
    return _reader


def get_ocr_error():
    return _reader_error


def warm_up():
    """Forces EasyOCR to load right now, in whatever thread calls this,
    instead of lazily on first real use (avoids a stutter the first
    time a twin-group product is encountered during live detection)."""
    _get_reader()


def get_oriented_crop(frame, polygon, pad=12):
    """Uses the segmentation polygon to extract an axis-aligned crop of
    just the product, rotated so its long axis is horizontal. Works
    off the mask we already predict -- no new labels/geometry needed."""
    pts = np.array(polygon, dtype=np.float32)
    if pts.shape[0] < 3:
        return None

    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    if w < 1 or h < 1:
        return None

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    x1 = int(cx - w / 2 - pad)
    y1 = int(cy - h / 2 - pad)
    x2 = int(cx + w / 2 + pad)
    y2 = int(cy + h / 2 + pad)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(rotated.shape[1], x2)
    y2 = min(rotated.shape[0], y2)

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _run_ocr(reader, img):
    """Runs OCR once, returns (concatenated_text, mean_confidence)."""
    try:
        results = reader.readtext(img)
    except Exception:
        return "", 0.0

    if not results:
        return "", 0.0

    texts = [r[1] for r in results]
    confs = [r[2] for r in results]
    combined_text = " ".join(texts)
    mean_conf = sum(confs) / len(confs)
    return combined_text, mean_conf


def best_rotation_ocr(crop):
    """Tries the crop at 0/90/180/270 degrees (since minAreaRect's angle
    has a quadrant ambiguity) and keeps whichever rotation gives OCR
    the highest mean confidence. Returns (text, confidence)."""
    reader = _get_reader()
    if reader is None or crop is None:
        return "", 0.0

    best_text, best_conf = "", 0.0
    rotations = [
        crop,
        cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(crop, cv2.ROTATE_180),
        cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    for rot in rotations:
        text, conf = _run_ocr(reader, rot)
        if conf > best_conf:
            best_text, best_conf = text, conf

    return best_text, best_conf


def _normalize(text):
    """Uppercase, strip everything but alphanumerics - makes OCR text
    robust to spacing/punctuation noise before matching against known
    part codes."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def match_text_to_product(ocr_text, candidates):
    """Fuzzy-matches OCR output against a set of candidate product
    labels. Returns the best-matching candidate, or None if nothing is
    a plausible match. Uses normalized substring containment first
    (cheap, robust to OCR noise around the edges), falling back to
    longest-common-substring scoring for partial/noisy reads."""
    if not ocr_text:
        return None

    norm_text = _normalize(ocr_text)
    if not norm_text:
        return None

    candidates = list(candidates)

    # 1. Exact substring containment (most reliable when OCR gets most
    #    of the code right).
    for candidate in candidates:
        norm_candidate = _normalize(candidate)
        if norm_candidate in norm_text or norm_text in norm_candidate:
            return candidate

    # 2. Fallback: longest-common-substring scoring for partial/noisy
    #    reads (handles cases where OCR drops/garbles a character or
    #    two but gets most of the code right).
    best_candidate, best_score = None, 0
    for candidate in candidates:
        norm_candidate = _normalize(candidate)
        score = _lcs_length(norm_text, norm_candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None:
        return None

    # Require a reasonably long common substring before trusting it -
    # avoids false positives on very short/garbled OCR reads. Scaled
    # to the SHORTEST candidate in this specific group, since some
    # twin groups have codes of quite different lengths (e.g. the
    # EC/EP/CCB group).
    min_len = min(len(_normalize(c)) for c in candidates)
    if best_score >= max(6, int(min_len * 0.6)):
        return best_candidate

    return None


def _lcs_length(a, b):
    """Longest common substring length between two strings."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
    return best
