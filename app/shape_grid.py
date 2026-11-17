"""Recognition and solving of the odd-one-out shape grid made by generator.py.

A 3 x 3 grid holds eight tiles with the same shape and one tile with a
different shape. The answer is the number of the odd tile ("1" to "9",
left to right, top to bottom).

Primary method : classify each tile's shape from contour geometry and pick
                 the tile whose class differs from the other eight.
Fallback method: when the classes do not give exactly one odd tile, pick the
                 tile whose geometry is farthest from the grid's median.
"""
from __future__ import annotations

import cv2
import numpy as np

CELLS = 3

# Geometry of each shape: (solidity, extent, circularity), where
#   solidity    = area / convex-hull area
#   extent      = area / minimum-area bounding rectangle
#   circularity = 4 * pi * area / perimeter^2
# All three are scale- and rotation-invariant. The values are the means
# measured on about 5,400 tiles from 600 generated grids.
PROTOTYPES = {
    "circle": (0.987, 0.790, 0.897),
    "square": (0.965, 0.960, 0.697),
    "triangle": (0.954, 0.491, 0.533),
    "star": (0.508, 0.353, 0.278),
}
_NAMES = list(PROTOTYPES)
_PROTO = np.array([PROTOTYPES[n] for n in _NAMES])
_SHARPNESS = 60.0  # turns squared feature distance into a class probability


def looks_like_grid(gray: np.ndarray) -> bool:
    """True when grey separator lines sit at one third and two thirds of both axes."""
    h, w = gray.shape
    if min(h, w) < 150 or not 0.9 <= w / h <= 1.1:
        return False
    not_background = gray < 215
    if not 0.03 <= not_background.mean() <= 0.5:  # shapes and lines on a light background
        return False
    for profile, size in ((not_background.mean(axis=0), w), (not_background.mean(axis=1), h)):
        tol = max(2, int(size * 0.03))
        for k in range(1, CELLS):
            pos = round(size * k / CELLS)
            if profile[max(0, pos - tol):pos + tol + 1].max() < 0.8:
                return False
    return True


def tile_features(binary: np.ndarray) -> list[np.ndarray | None]:
    """Geometry of the largest blob in each tile, or None for an empty tile."""
    h, w = binary.shape
    ch, cw = h / CELLS, w / CELLS
    inset = max(2, int(min(ch, cw) * 0.04))
    min_area = 0.03 * ch * cw
    feats: list[np.ndarray | None] = []
    for i in range(CELLS * CELLS):
        r, c = divmod(i, CELLS)
        tile = binary[int(r * ch) + inset:int((r + 1) * ch) - inset,
                      int(c * cw) + inset:int((c + 1) * cw) - inset]
        contours, _ = cv2.findContours(tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best = max(contours, key=cv2.contourArea, default=None)
        area = cv2.contourArea(best) if best is not None else 0.0
        if area < min_area:
            feats.append(None)
            continue
        hull = cv2.contourArea(cv2.convexHull(best))
        (_, (rw, rh), _) = cv2.minAreaRect(best)
        perimeter = cv2.arcLength(best, True)
        feats.append(np.array([area / max(hull, 1.0), area / max(rw * rh, 1.0),
                               4 * np.pi * area / max(perimeter ** 2, 1.0)]))
    return feats


def is_shape_grid(gray: np.ndarray, binary: np.ndarray) -> bool:
    return looks_like_grid(gray) and all(f is not None for f in tile_features(binary))


def solve(binary: np.ndarray) -> tuple[str | None, float | None, bool, str]:
    """Return (solution, confidence, fallback_used, detail)."""
    feats = tile_features(binary)
    if any(f is None for f in feats):
        return None, None, False, "at least one tile holds no shape"
    X = np.stack(feats)
    dist = ((X[:, None, :] - _PROTO[None, :, :]) ** 2).sum(axis=2)
    probs = np.exp(-_SHARPNESS * dist)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = probs.argmax(axis=1)
    certainty = float(probs.max(axis=1).mean())

    counts = np.bincount(labels, minlength=len(_NAMES))
    common = int(counts.argmax())
    odd = np.flatnonzero(labels != common)
    if len(odd) == 1:
        i = int(odd[0])
        detail = (f"tile {i + 1} (row {i // CELLS + 1}, column {i % CELLS + 1}) holds a "
                  f"{_NAMES[labels[i]]} among {_NAMES[common]}s")
        return str(i + 1), round(certainty, 4), False, detail

    # Fallback: the classes disagree, so use the geometric outlier instead.
    spread = ((X - np.median(X, axis=0)) ** 2).sum(axis=1)
    i = int(spread.argmax())
    ranked = np.sort(spread)
    margin = float(1 - ranked[-2] / ranked[-1]) if ranked[-1] > 0 else 0.0
    detail = (f"shape classes gave {len(odd)} odd tiles; picked tile {i + 1} as the "
              f"geometric outlier; low confidence")
    return str(i + 1), round(0.5 * certainty * margin, 4), True, detail
