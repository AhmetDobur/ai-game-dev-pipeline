"""Dominant colours of a reference image, used to tint scaffold-authored geometry.

The generated props carry their own baked textures, but the room around them --
floor, walls, fog, ambient -- is authored by scaffold.py and would otherwise be
hardcoded to whatever looked right for one game. Reading the palette off the
run's own environment reference keeps the built room in the same key as the art.
No model involved: PIL's median-cut quantiser is enough and costs milliseconds.
"""
from pathlib import Path

# Fallback used when there is no environment reference (or it fails to open):
# the previous hardcoded warm-dark scheme, so behaviour is unchanged without one.
DEFAULT = [(0.06, 0.05, 0.04), (0.23, 0.17, 0.12), (0.35, 0.11, 0.12),
           (0.72, 0.55, 0.22), (0.55, 0.50, 0.45)]


def extract(image: str | Path, colors: int = 6) -> list[tuple[float, float, float]]:
    """Dominant colours as 0..1 RGB triples, most-common first."""
    try:
        from PIL import Image
        with Image.open(image) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            q = im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
            pal = q.getpalette()[: colors * 3]
            counts = sorted(q.getcolors() or [], reverse=True)
    except Exception:  # a reference we cannot read must never fail a build
        return list(DEFAULT)
    out = []
    for _, idx in counts:
        r, g, b = pal[idx * 3: idx * 3 + 3]
        out.append((round(r / 255, 3), round(g / 255, 3), round(b / 255, 3)))
    return out or list(DEFAULT)


def _lum(c: tuple[float, float, float]) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def roles(image: str | Path | None) -> dict:
    """Map a palette onto the room's material slots by luminance and saturation."""
    pal = extract(image) if image else list(DEFAULT)
    by_lum = sorted(pal, key=_lum)
    darkest, brightest = by_lum[0], by_lum[-1]
    # the most saturated mid-tone reads as the room's accent (oxblood, in a
    # baroque library); gold/brass is simply the brightest warm swatch
    def sat(c):
        return max(c) - min(c)
    accent = max(by_lum[: max(2, len(by_lum) - 1)], key=sat)
    return {
        "shadow": darkest,
        "wood": by_lum[len(by_lum) // 3],
        "accent": accent,
        "gold": brightest,
        "stone": by_lum[len(by_lum) // 2],
    }
