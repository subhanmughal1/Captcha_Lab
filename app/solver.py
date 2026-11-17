"""Custom solving engine for the synthetic CAPTCHAs made by generator.py.

Two challenge types are supported:

* synthetic_text        text strip, 4 to 6 characters (this module)
* synthetic_shape_grid  3 x 3 odd-one-out shape grid (shape_grid.py)

Text pipeline: OpenCV clean-up -> length estimate -> segmentation -> two small
neural-network classifiers (scikit-learn MLP) averaged -> solution + confidence.

Primary method : layout-guided segmentation for the estimated length, read by
                 the ensemble of both classifiers.
Fallback method: when the primary result is not confident, the other lengths
                 and a connected-component segmentation are tried and the
                 best-scoring reading is returned.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import joblib
import numpy as np

from . import shape_grid

MODEL_PATH = Path(__file__).parent.parent / "models" / "char_mlp.joblib"
MODEL_VERSION = 2
CHAR_SIZE = 28
WINDOW_SIZE = (24, 36)          # (width, height) of the fixed slot window
CONFIDENCE_THRESHOLD = 0.60
LENGTH_THRESHOLD = 0.60         # minimum certainty of the length estimate
EXPECTED_LENGTHS = range(4, 7)

# Layout of the text generator, relative to a 220 px wide image: a 10 px margin
# on each side and one equal slot per character in between.
REF_WIDTH = 220
REF_MARGIN = 10

_model = None


class UnsupportedChallenge(Exception):
    """Raised when the input is not a challenge type this engine supports."""


@dataclass
class SolveResult:
    challenge_type: str
    supported: bool
    solution: str | None
    confidence: float | None
    method: str
    fallback_used: bool
    processing_ms: float
    est_cost_usd: float = 0.0  # custom local engine: no per-challenge fee
    detail: str = ""

    def dict(self) -> dict:
        return asdict(self)


def load_model() -> dict:
    """Return {"length": ..., "glyph": ..., "window": ...} (see scripts/train.py)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model file missing. Run: python scripts/train.py")
        try:
            loaded = joblib.load(MODEL_PATH)
        except Exception as exc:  # e.g. trained with a different scikit-learn version
            raise RuntimeError("Model could not be loaded. Run: python scripts/train.py") from exc
        if not isinstance(loaded, dict) or loaded.get("version") != MODEL_VERSION:
            raise RuntimeError("Model file is from an older version. Run: python scripts/train.py")
        _model = loaded
    return _model


# ---------- image handling ----------

def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise UnsupportedChallenge("File is not a readable image")
    return img


def binarize(img: np.ndarray) -> np.ndarray:
    """Dark ink -> white on black. Grey noise lines and speckle are dropped
    by a fixed darkness threshold plus a small-component filter."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    # Noise lines are drawn over the glyphs and cut 1px gaps into them: bridge those.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Drop speckle by component area instead of erosion, so thin serif strokes survive.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < 12]
    if small:
        binary[np.isin(labels, small)] = 0
    return binary


def _normalise(crop: np.ndarray) -> np.ndarray:
    """Glyph feature: the crop's ink, scaled to fit and centred in 28 x 28."""
    ys, xs = np.nonzero(crop)
    if len(xs) == 0:
        return np.zeros(CHAR_SIZE * CHAR_SIZE, np.float32)
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 22 / max(h, w)
    crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CHAR_SIZE, CHAR_SIZE), np.uint8)
    h, w = crop.shape
    y0, x0 = (CHAR_SIZE - h) // 2, (CHAR_SIZE - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    return (canvas.astype(np.float32) / 255).ravel()


# ---------- segmentation ----------

def segment_contours(binary: np.ndarray) -> list[np.ndarray]:
    """Connected components, merged when they overlap horizontally (keeps
    multi-part glyphs together), ordered left to right. Used for challenge-type
    recognition and as one of the fallback readings."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = [[x, y, x + w, y + h] for x, y, w, h, area in stats[1:] if area >= 25]
    boxes.sort()
    merged: list[list[int]] = []
    for b in boxes:
        if merged:
            m = merged[-1]
            overlap = min(m[2], b[2]) - b[0]
            # Same glyph only if the boxes share most of the smaller one's width;
            # neighbouring glyphs that merely touch stay separate.
            if overlap > 0.5 * min(m[2] - m[0], b[2] - b[0]):
                merged[-1] = [m[0], min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
                continue
        merged.append(b)
    merged = [b for b in merged if (b[3] - b[1]) >= 14]
    crops = []
    for x0, y0, x1, y1 in merged:
        crop = binary[y0:y1, x0:x1]
        if (x1 - x0) > 46:  # two glyphs joined by ink: split at the thinnest column
            mid = crop[:, 12:-12].sum(axis=0)
            cut = 12 + int(mid.argmin())
            crops += [crop[:, :cut], crop[:, cut:]]
        else:
            crops.append(crop)
    return crops


def _slot_width(width: int, count: int) -> tuple[float, float]:
    """(scale, slot width in px) of the generator layout for `count` characters."""
    scale = width / REF_WIDTH
    return scale, ((REF_WIDTH - 2 * REF_MARGIN) // count) * scale


def segment_layout(binary: np.ndarray, count: int) -> list[np.ndarray]:
    """Cut the strip into `count` glyph crops. Each cut starts at the slot
    boundary the generator layout predicts and moves, within a few pixels, to
    the column with the least ink, so touching glyphs are separated cleanly."""
    w = binary.shape[1]
    scale, slot = _slot_width(w, count)
    ink = binary.sum(axis=0).astype(np.float64)
    reach = 7 * scale
    cuts = [0]
    for i in range(1, count):
        nominal = REF_MARGIN * scale + i * slot
        lo, hi = int(max(1, nominal - reach)), int(min(w - 1, nominal + reach))
        cols = np.arange(lo, hi + 1)
        cuts.append(int(cols[(ink[lo:hi + 1] + 0.01 * np.abs(cols - nominal)).argmin()]))
    cuts.append(w)
    return [binary[:, cuts[i]:cuts[i + 1]] for i in range(count)]


def slot_windows(binary: np.ndarray, count: int) -> list[np.ndarray]:
    """Window feature: a fixed-width view centred on each slot. It keeps the
    whole glyph even when a neighbour overlaps it, which a hard cut cannot."""
    w = binary.shape[1]
    scale, slot = _slot_width(w, count)
    half = 22 * scale
    pad = int(half) + 2
    padded = cv2.copyMakeBorder(binary, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    views = []
    for i in range(count):
        centre = REF_MARGIN * scale + (i + 0.5) * slot
        x0 = int(round(centre - half)) + pad
        view = padded[:, x0:x0 + max(2, int(round(2 * half)))]
        view = cv2.resize(view, WINDOW_SIZE, interpolation=cv2.INTER_AREA)
        views.append((view.astype(np.float32) / 255).ravel())
    return views


def length_feature(binary: np.ndarray) -> np.ndarray:
    """Column ink profile resampled to 110 bins: the valleys between glyphs
    reveal how many characters the strip holds."""
    profile = binary.sum(axis=0).astype(np.float32) / (255 * binary.shape[0])
    return np.interp(np.linspace(0, len(profile) - 1, 110), np.arange(len(profile)), profile)


# ---------- reading ----------

def _read_layout(binary: np.ndarray, count: int) -> tuple[str, float]:
    """Read `count` characters with both classifiers and average them."""
    model = load_model()
    glyph = model["glyph"].predict_proba(
        np.stack([_normalise(c) for c in segment_layout(binary, count)]))
    window = model["window"].predict_proba(np.stack(slot_windows(binary, count)))
    probs = (glyph + window) / 2
    text = "".join(model["glyph"].classes_[probs.argmax(axis=1)])
    return text, float(probs.max(axis=1).mean())


def _read_contours(chars: list[np.ndarray]) -> tuple[str, float]:
    model = load_model()["glyph"]
    probs = model.predict_proba(np.stack([_normalise(c) for c in chars]))
    return "".join(model.classes_[probs.argmax(axis=1)]), float(probs.max(axis=1).mean())


def _solve_text(binary: np.ndarray) -> tuple[str, float, str, bool, str]:
    """Return (solution, confidence, method, fallback_used, note)."""
    length_model = load_model()["length"]
    length_probs = dict(zip((int(c) for c in length_model.classes_),
                            length_model.predict_proba([length_feature(binary)])[0]))
    count = max(length_probs, key=length_probs.get)
    text, conf = _read_layout(binary, count)
    if length_probs[count] >= LENGTH_THRESHOLD and conf >= CONFIDENCE_THRESHOLD:
        return text, conf, "custom_cv_mlp (layout + ensemble)", False, ""

    # Fallback: score every plausible reading by character confidence weighted
    # with how likely its length is, and keep the best one.
    candidates = [(conf * length_probs[count], text, conf, "custom_cv_mlp (layout + ensemble)", False)]
    for other in EXPECTED_LENGTHS:
        if other != count:
            t, c = _read_layout(binary, other)
            candidates.append((c * length_probs.get(other, 0.0), t, c,
                               "custom_cv_mlp (alternative length)", True))
    chars = segment_contours(binary)
    if len(chars) in EXPECTED_LENGTHS:
        t, c = _read_contours(chars)
        candidates.append((c * length_probs.get(len(chars), 0.0), t, c,
                           "custom_cv_mlp (contours)", True))
    _, text, conf, method, used_fallback = max(candidates, key=lambda cand: cand[0])
    note = "; low confidence" if conf < CONFIDENCE_THRESHOLD else ""
    return text, conf, method, used_fallback, note


# ---------- challenge-type recognition ----------

def classify_challenge(img: np.ndarray, binary: np.ndarray) -> tuple[str, str]:
    """Return (challenge_type, reason). Supported types are 'synthetic_text'
    and 'synthetic_shape_grid'; every other value starts with 'unsupported'."""
    h, w = img.shape[:2]
    if w < 80 or h < 30:
        return "unsupported", "image too small to be a supported challenge"
    if shape_grid.is_shape_grid(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), binary):
        return "synthetic_shape_grid", "3 x 3 grid with one shape in every tile"
    if w / h < 1.8:
        return "unsupported_image", "not a text strip and not a 3 x 3 shape grid"
    ink = np.count_nonzero(binary) / binary.size
    if ink < 0.01 or ink > 0.45:
        return "unsupported", "ink ratio is outside the range of a text challenge"
    count = len(segment_contours(binary))
    if count not in range(2, 10):
        return "unsupported", f"found {count} glyph-like regions"
    return "synthetic_text", f"{count} glyph-like regions on a plain background"


# ---------- public entry point ----------

def solve(data: bytes) -> SolveResult:
    start = time.perf_counter()

    def done(**kw) -> SolveResult:
        return SolveResult(processing_ms=round((time.perf_counter() - start) * 1000, 2), **kw)

    try:
        img = decode(data)
    except UnsupportedChallenge as exc:
        return done(challenge_type="invalid_input", supported=False, solution=None,
                    confidence=None, method="none", fallback_used=False, detail=str(exc))

    binary = binarize(img)
    ctype, reason = classify_challenge(img, binary)

    if ctype == "synthetic_shape_grid":
        solution, conf, used_fallback, detail = shape_grid.solve(binary)
        return done(challenge_type=ctype, supported=True, solution=solution, confidence=conf,
                    method="custom_cv_shapes (outlier)" if used_fallback else "custom_cv_shapes (geometry)",
                    fallback_used=used_fallback, detail=detail)

    if ctype != "synthetic_text":
        return done(challenge_type=ctype, supported=False, solution=None, confidence=None,
                    method="none", fallback_used=False, detail=reason)

    text, conf, method, used_fallback, note = _solve_text(binary)
    return done(challenge_type=ctype, supported=True, solution=text, confidence=round(conf, 4),
                method=method, fallback_used=used_fallback, detail=reason + note)
