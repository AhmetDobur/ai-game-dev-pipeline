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
    """Foreground via rembg's u2net matte when available (robust on textured or
    smoky backgrounds), else border-color distance (uniform backgrounds only)."""
    w, h = img.size
    try:
        from rembg import remove
        matte = remove(img).getchannel("A").load()
        # strict threshold: wispy props (chains, veils) that bridge figures
        # sit in the matte's soft range and must not connect components
        return [[matte[x, y] > 200 for x in range(w)] for y in range(h)]
    except Exception:
        pass  # rembg missing or model fetch failed -> heuristic fallback
    px = img.load()
    bg = _background_color(px, w, h)
    return [[max(abs(px[x, y][0] - bg[0]), abs(px[x, y][1] - bg[1]),
                 abs(px[x, y][2] - bg[2])) > _BG_TOLERANCE
             for x in range(w)] for y in range(h)]


def _erode(mask: list[list[bool]], passes: int = 2) -> list[list[bool]]:
    """4-neighborhood erosion: breaks the thin bridges (floor shadows, chains,
    contact points) that fuse separate figures into one component. Bodies are
    thick at analysis scale and survive."""
    h, w = len(mask), len(mask[0])
    for _ in range(passes):
        nxt = [[mask[y][x]
                and (y > 0 and mask[y - 1][x]) and (y < h - 1 and mask[y + 1][x])
                and (x > 0 and mask[y][x - 1]) and (x < w - 1 and mask[y][x + 1])
                for x in range(w)] for y in range(h)]
        mask = nxt
    return mask


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


def _xy_cut(mask, x0: int, y0: int, x1: int, y1: int, depth: int = 4) -> list:
    """Recursive XY-cut: split the region at its widest empty row/column band
    (near-zero foreground), the classic layout segmentation. Character sheets
    are grids of figures; connected-component analysis alone can't separate
    figures that touch through props, but the gaps between grid cells can."""
    if depth == 0 or x1 - x0 < 8 or y1 - y0 < 8:
        return [(x0, y0, x1, y1)]
    col = [sum(mask[y][x] for y in range(y0, y1)) for x in range(x0, x1)]
    row = [sum(mask[y][x] for x in range(x0, x1)) for y in range(y0, y1)]

    def widest_gap(profile, span):
        limit = max(1, int(0.02 * span))
        best, cur_start, i0, i1 = 0, None, 0, 0
        for i, v in enumerate(profile + [limit + 1]):
            if v <= limit:
                if cur_start is None:
                    cur_start = i
            elif cur_start is not None:
                if i - cur_start > best:
                    best, i0, i1 = i - cur_start, cur_start, i
                cur_start = None
        return best, i0, i1

    cg, cs, ce = widest_gap(col, y1 - y0)
    rg, rs, re_ = widest_gap(row, x1 - x0)
    # a real separator is interior and at least 3px wide
    cut_col = cg >= 3 and 0 < cs and ce < (x1 - x0)
    cut_row = rg >= 3 and 0 < rs and re_ < (y1 - y0)
    if cut_col and (cg >= rg or not cut_row):
        mid = x0 + (cs + ce) // 2
        return (_xy_cut(mask, x0, y0, mid, y1, depth - 1)
                + _xy_cut(mask, mid, y0, x1, y1, depth - 1))
    if cut_row:
        mid = y0 + (rs + re_) // 2
        return (_xy_cut(mask, x0, y0, x1, mid, depth - 1)
                + _xy_cut(mask, x0, mid, x1, y1, depth - 1))
    return [(x0, y0, x1, y1)]


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
    erode_px = 4   # breaks prop bridges; bbox is re-expanded below
    mask = _erode(_foreground_mask(small), erode_px)
    sw, sh = small.size
    # XY-cut the sheet into layout cells, keep the densest cell, then take the
    # largest connected figure inside it
    cells = _xy_cut(mask, 0, 0, sw, sh)
    def density(c):
        return sum(mask[y][x] for y in range(c[1], c[3]) for x in range(c[0], c[2]))
    cx0, cy0, cx1, cy1 = max(cells, key=density)
    cell_mask = [[mask[y][x] if cx0 <= x < cx1 and cy0 <= y < cy1 else False
                  for x in range(sw)] for y in range(sh)]
    size, x0, y0, x1, y1 = _largest_component(cell_mask)
    if size == 0:
        return src  # nothing detected — don't guess
    x0, y0 = max(0, x0 - erode_px), max(0, y0 - erode_px)     # undo erosion shrink
    x1, y1 = min(sw - 1, x1 + erode_px), min(sh - 1, y1 + erode_px)
    blob_w, blob_h = x1 - x0 + 1, y1 - y0 + 1
    if blob_w >= _FULL_FRAME * sw and blob_h >= _FULL_FRAME * sh:
        return src  # one subject already fills the frame — cropping adds nothing
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
