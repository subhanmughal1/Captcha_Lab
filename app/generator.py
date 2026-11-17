"""Synthetic CAPTCHA generator: text strips and odd-one-out shape grids.

Everything this project solves is produced here (or uploaded by the user from
this same generator). It is a candidate-controlled dataset, as required by the
assignment scope.
"""
from __future__ import annotations

import io
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).parent / "assets"
FONTS = [ASSETS / "DejaVuSans-Bold.ttf", ASSETS / "DejaVuSerif-Bold.ttf"]

# Ambiguous glyphs (0/O, 1/I) are left out on purpose.
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
WIDTH, HEIGHT = 220, 80


def random_text(length: int = 5, rng: random.Random | None = None) -> str:
    rng = rng or random
    return "".join(rng.choice(ALPHABET) for _ in range(length))


def generate(text: str | None = None, length: int = 5,
             rng: random.Random | None = None) -> tuple[str, Image.Image]:
    """Return (answer, image). Each character gets its own slot, a random
    rotation, font, size and colour; thin lines and dots are added as noise."""
    rng = rng or random
    text = text or random_text(length, rng)
    img = Image.new("RGB", (WIDTH, HEIGHT),
                    (rng.randint(235, 255), rng.randint(235, 255), rng.randint(235, 255)))
    slot = (WIDTH - 20) // len(text)

    for i, ch in enumerate(text):
        font = ImageFont.truetype(str(rng.choice(FONTS)), rng.randint(34, 42))
        glyph = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
        colour = (rng.randint(0, 75), rng.randint(0, 75), rng.randint(0, 75), 255)
        ImageDraw.Draw(glyph).text((30, 30), ch, font=font, fill=colour, anchor="mm")
        glyph = glyph.rotate(rng.uniform(-18, 18), resample=Image.BICUBIC, expand=False)
        x = 10 + i * slot + (slot - 60) // 2 + rng.randint(-3, 3)
        y = (HEIGHT - 60) // 2 + rng.randint(-6, 6)
        img.paste(glyph, (x, y), glyph)

    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 6)):  # thin grey noise lines
        g = rng.randint(125, 190)
        draw.line([(rng.randint(0, WIDTH), rng.randint(0, HEIGHT)),
                   (rng.randint(0, WIDTH), rng.randint(0, HEIGHT))], fill=(g, g, g), width=1)
    for _ in range(rng.randint(60, 140)):  # speckle
        g = rng.randint(125, 200)
        draw.point((rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)), fill=(g, g, g))
    return text, img


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- second challenge type: odd-one-out shape grid ----------

GRID_SIZE = 300          # square image
GRID_CELLS = 3           # 3 x 3 tiles
SHAPES = ("circle", "square", "triangle", "star")


def _shape_points(shape: str, cx: float, cy: float, r: float, angle: float) -> list[tuple[float, float]]:
    """Polygon vertices for a shape of radius r, rotated by `angle` radians."""
    if shape == "square":
        steps, radii = 4, [r]
    elif shape == "triangle":
        steps, radii = 3, [r * 1.12]
    else:  # star: 5 outer points with an inner radius
        steps, radii = 10, [r * 1.12, r * 0.47]
    return [(cx + radii[k % len(radii)] * math.cos(angle + 2 * math.pi * k / steps),
             cy + radii[k % len(radii)] * math.sin(angle + 2 * math.pi * k / steps))
            for k in range(steps)]


def generate_shape_grid(rng: random.Random | None = None) -> tuple[str, Image.Image]:
    """Return (answer, image) for an odd-one-out challenge: eight tiles hold the
    same shape and one tile holds a different one. The answer is the number of
    the odd tile, "1" to "9", counted left to right and top to bottom."""
    rng = rng or random
    common, odd = rng.sample(SHAPES, 2)
    odd_index = rng.randrange(GRID_CELLS * GRID_CELLS)
    img = Image.new("RGB", (GRID_SIZE, GRID_SIZE),
                    (rng.randint(235, 255), rng.randint(235, 255), rng.randint(235, 255)))
    draw = ImageDraw.Draw(img)
    cell = GRID_SIZE // GRID_CELLS

    for k in range(1, GRID_CELLS):  # tile borders (mid grey, so they are not read as ink)
        g = rng.randint(150, 175)
        draw.line([(k * cell, 0), (k * cell, GRID_SIZE)], fill=(g, g, g), width=2)
        draw.line([(0, k * cell), (GRID_SIZE, k * cell)], fill=(g, g, g), width=2)

    for i in range(GRID_CELLS * GRID_CELLS):
        shape = odd if i == odd_index else common
        cx = (i % GRID_CELLS) * cell + cell / 2 + rng.randint(-7, 7)
        cy = (i // GRID_CELLS) * cell + cell / 2 + rng.randint(-7, 7)
        r = rng.randint(24, 33)
        colour = (rng.randint(0, 90), rng.randint(0, 90), rng.randint(0, 90))
        if shape == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
        else:
            draw.polygon(_shape_points(shape, cx, cy, r, rng.uniform(0, 2 * math.pi)), fill=colour)

    for _ in range(rng.randint(3, 6)):  # thin grey noise lines
        g = rng.randint(125, 190)
        draw.line([(rng.randint(0, GRID_SIZE), rng.randint(0, GRID_SIZE)),
                   (rng.randint(0, GRID_SIZE), rng.randint(0, GRID_SIZE))], fill=(g, g, g), width=1)
    for _ in range(rng.randint(120, 260)):  # speckle
        g = rng.randint(125, 200)
        draw.point((rng.randint(0, GRID_SIZE - 1), rng.randint(0, GRID_SIZE - 1)), fill=(g, g, g))
    return str(odd_index + 1), img


KINDS = ("text", "shape_grid")


def generate_kind(kind: str, length: int = 5,
                  rng: random.Random | None = None) -> tuple[str, Image.Image]:
    """Generate one challenge of the given kind ("text" or "shape_grid")."""
    if kind == "shape_grid":
        return generate_shape_grid(rng)
    return generate(length=length, rng=rng)
