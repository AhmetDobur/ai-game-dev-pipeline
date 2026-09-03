"""Reference-image preparation: crop a multi-figure sheet to its main subject.

TRELLIS reconstructs everything in frame, so a character sheet with turnarounds
becomes N meshes. Before conditioning 3D generation on a user reference we crop
it to the single largest figure. Pure Pillow — no numpy/opencv on the box.
"""
from pathlib import Path

from PIL import Image

_SCALE = 192          # analysis resolution; components found here, crop on original
_BG_TOLERANCE = 34    # per-channel distance from the border-median background color
_FULL_FRAME = 0.80    # largest blob covers this much -> already single-subject


def _background_color(px, w: int, h: int) -> tuple[int, int, int]:
    """Median of the border pixels — character sheets have uniform backgrounds."""
    border = ([px[x, 0] for x in range(w)] + [px[x, h - 1] for x in range(w)]
              + [px[0, y] for y in range(h)] + [px[w - 1, y] for y in range(h)])
    rs, gs, bs = (sorted(c[i] for c in border) for i in range(3))
    mid = len(border) // 2
    return rs[mid], gs[mid], bs[mid]


def _foreground_mask(img: Image.Image) -> list[list[bool]]:
    px = img.load()
    w, h = img.size
    bg = _background_color(px, w, h)
    return [[max(abs(px[x, y][0] - bg[0]), abs(px[x, y][1] - bg[1]),
                 abs(px[x, y][2] - bg[2])) > _BG_TOLERANCE
             for x in range(w)] for y in range(h)]


def _largest_component(mask: list[list[bool]]) -> tuple[int, int, int, int, int]:
    """Iterative 4-connected flood fill. Returns (size, x0, y0, x1, y1) of the
    largest foreground blob. Small: analysis image is <=192px a side."""
    h, w = len(mask), len(mask[0])
    seen = [[False] * w for _ in range(h)]
    best = (0, 0, 0, 0, 0)
    for sy in range(h):
        for sx in range(w):
            if not mask[sy][sx] or seen[sy][sx]:
                continue
            stack, size = [(sx, sy)], 0
            x0, y0, x1, y1 = sx, sy, sx, sy
            seen[sy][sx] = True
            while stack:
                x, y = stack.pop()
                size += 1
                x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < w and 0 <= ny < h and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if size > best[0]:
                best = (size, x0, y0, x1, y1)
    return best


def crop_main_subject(src: Path, dst: Path) -> Path:
    """Write a crop of src's largest connected figure to dst (PNG). Falls back to
    the original image whenever analysis is inconclusive — never blocks a run."""
    try:
        img = Image.open(src).convert("RGB")
    except OSError:
        return src
    w, h = img.size
    scale = max(w, h) / _SCALE
    small = img.resize((max(1, round(w / scale)), max(1, round(h / scale)))) \
        if scale > 1 else img.copy()
    mask = _foreground_mask(small)
    size, x0, y0, x1, y1 = _largest_component(mask)
    sw, sh = small.size
    if size == 0 or size >= _FULL_FRAME * sw * sh:
        return src  # nothing detected, or one subject already fills the frame
    blob_w, blob_h = x1 - x0 + 1, y1 - y0 + 1
    if blob_w * blob_h >= _FULL_FRAME * sw * sh:
        return src  # blob spans the sheet (grid of figures merged) — don't guess
    f = max(w, h) / max(sw, sh)
    margin = 0.08
    cx0 = max(0, int((x0 - blob_w * margin) * f))
    cy0 = max(0, int((y0 - blob_h * margin) * f))
    cx1 = min(w, int((x1 + 1 + blob_w * margin) * f))
    cy1 = min(h, int((y1 + 1 + blob_h * margin) * f))
    crop = img.crop((cx0, cy0, cx1, cy1))
    # square canvas in the sheet's own background color: TRELLIS conditioning
    # expects a centered subject, not an off-aspect sliver
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), _background_color(img.load(), w, h))
    canvas.paste(crop, ((side - crop.size[0]) // 2, (side - crop.size[1]) // 2))
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, "PNG")
    return dst
