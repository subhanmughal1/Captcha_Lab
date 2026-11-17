"""Train the text-CAPTCHA models on freshly generated synthetic CAPTCHAs.

Three small scikit-learn MLPs are trained and saved together:

* length  how many characters the strip holds (from the column ink profile)
* glyph   one character from its cropped, centred ink (28 x 28)
* window  one character from a fixed window centred on its slot (24 x 36)

Usage:  python scripts/train.py [num_captchas]      (default 12000, a few minutes)
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from app import generator, solver


def build_dataset(n: int, seed: int):
    rng = random.Random(seed)
    length_x, length_y = [], []
    glyph_x, glyph_y = [], []
    window_x, window_y = [], []
    for _ in range(n):
        text, img = generator.generate(length=rng.choice(list(solver.EXPECTED_LENGTHS)), rng=rng)
        binary = solver.binarize(solver.decode(generator.to_png_bytes(img)))
        length_x.append(solver.length_feature(binary))
        length_y.append(len(text))
        for ch, crop in zip(text, solver.segment_layout(binary, len(text))):
            glyph_x.append(solver._normalise(crop))
            glyph_y.append(ch)
        chars = solver.segment_contours(binary)
        if len(chars) == len(text):  # also teach the fallback's crops, when they are clean
            for ch, crop in zip(text, chars):
                glyph_x.append(solver._normalise(crop))
                glyph_y.append(ch)
        for ch, view in zip(text, solver.slot_windows(binary, len(text))):
            window_x.append(view)
            window_y.append(ch)
    return ((np.stack(length_x), np.array(length_y)),
            (np.stack(glyph_x), np.array(glyph_y)),
            (np.stack(window_x), np.array(window_y)))


def fit(name: str, data, hidden) -> MLPClassifier:
    X, y = data
    start = time.time()
    model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=80,
                          early_stopping=True, random_state=0).fit(X, y)
    print(f"{name}: {len(X)} samples, {model.n_iter_} epochs, {time.time() - start:.0f} s")
    return model


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    print(f"generating {n} captchas ...")
    length_data, glyph_data, window_data = build_dataset(n, seed=1)
    bundle = {
        "version": solver.MODEL_VERSION,
        "length": fit("length", length_data, (128,)),
        "glyph": fit("glyph", glyph_data, (256, 128)),
        "window": fit("window", window_data, (256, 128)),
    }
    solver.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, solver.MODEL_PATH, compress=3)
    print("saved", solver.MODEL_PATH)


if __name__ == "__main__":
    main()
