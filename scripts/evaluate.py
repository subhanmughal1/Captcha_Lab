"""Measure solver accuracy on unseen synthetic CAPTCHAs.

Usage:  python scripts/evaluate.py [num_captchas] [text|shape_grid|mixed]
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import generator, solver


def evaluate(n: int = 500, seed: int = 999, kind: str = "text") -> dict:
    rng = random.Random(seed)
    correct = fallback = 0
    total_ms = 0.0
    by_type: dict[str, list[int]] = {}
    for _ in range(n):
        which = rng.choice(generator.KINDS) if kind == "mixed" else kind
        answer, img = generator.generate_kind(which, length=rng.choice([4, 5, 6]), rng=rng)
        res = solver.solve(generator.to_png_bytes(img))
        ok = res.solution == answer
        correct += ok
        fallback += res.fallback_used
        total_ms += res.processing_ms
        by_type.setdefault(res.challenge_type, []).append(ok)
    return {"total": n, "correct": correct, "accuracy": round(correct / n, 4),
            "fallback_used": fallback, "avg_ms": round(total_ms / n, 2),
            "accuracy_by_type": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_type.items())}}


if __name__ == "__main__":
    print(evaluate(int(sys.argv[1]) if len(sys.argv) > 1 else 500,
                   kind=sys.argv[2] if len(sys.argv) > 2 else "text"))
