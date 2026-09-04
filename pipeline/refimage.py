"""Reference-image preparation: crop a multi-figure sheet to one subject.

TRELLIS reconstructs everything in frame, so a character sheet with turnarounds
becomes N meshes. Before conditioning 3D generation on a user reference we crop
it to a single figure, selected by `index` — a two-hero sheet has two figures
worth reconstructing. Pure Pillow — no numpy/opencv on the box.
"""
from pathlib import Path

from PIL import Image

_SCALE = 192          # analysis resolution; components found here, crop on original
_BG_TOLERANCE = 34    # per-channel distance from the border-median background color
_FULL_FRAME = 0.80    # largest blob covers this much -> already single-subject
_ERODE_PX = 4         # breaks the prop bridges between touching figures
_MIN_RELATIVE_SIZE = 0.15   # a second hero is a sizeable fraction of the first
_MIN_FIGURE_ASPECT = 1.3    # standing characters are taller than they are wide
_MIN_MASK_COVERAGE = 0.05   # below this the matte failed; the frame IS the subject
_HEAD_BAND = 0.22           # top fraction of a standing figure: head + shoulders
_HEAD_UPSCALE = 1024        # a head crop is small; give the sampler pixels


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


def _components(mask: list[list[bool]]) -> list[tuple[int, int, int, int, int]]:
    """Every 4-connected blob as (size, x0, y0, x1, y1), largest first.

    A two-hero character sheet has two figures worth reconstructing, so the
    caller needs more than the winner. Small: analysis image is <=192px a side.
    """
    h, w = len(mask), len(mask[0])
    seen = [[False] * w for _ in range(h)]
    found = []
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
            found.append((size, x0, y0, x1, y1))
    found.sort(key=lambda c: -c[0])
    return found


def _subjects(mask: list[list[bool]]) -> list[tuple[int, int, int, int, int]]:
    """Figure-like blobs, largest first: big enough, and taller than wide.

    Replaces an XY-cut layout split that put a sheet's two heroes in different
    layout cells, so only ever one of them could be reached. The size threshold
    does the job the cell split was there for — rejecting the sheet's prop row —
    while keeping every real figure. Falls back to the largest blob when nothing
    is figure-shaped, so a prop or environment reference still crops as before.
    """
    comps = _components(mask)
    if not comps:
        return []
    big = comps[0][0]
    figures = [c for c in comps
               if c[0] >= _MIN_RELATIVE_SIZE * big
               and (c[4] - c[2] + 1) / max(c[3] - c[1] + 1, 1) >= _MIN_FIGURE_ASPECT]
    return figures or comps[:1]


def crop_head(src: Path, dst: Path, index: int = 0,
              band: float = _HEAD_BAND, upscale_to: int = _HEAD_UPSCALE) -> Path:
    """Write a head-and-shoulders close-up of one figure to dst (PNG).

    TRELLIS allocates voxels to what fills the frame, so a face that occupies
    3% of a full-body reference is reconstructed from almost no signal and comes
    back a blurred blob. Cropping to the head and upscaling gives the same
    generator the same budget for a face that a full-body crop gives a whole
    body. Falls back to the full-figure crop whenever the head cannot be located.
    """
    try:
        img = Image.open(src).convert("RGB")
    except Exception:
        return src
    w, h = img.size
    scale = max(w, h) / _SCALE
    small = img.resize((max(1, round(w / scale)), max(1, round(h / scale)))) \
        if scale > 1 else img.copy()
    mask = _erode(_foreground_mask(small), _ERODE_PX)
    sw, sh = small.size
    figures = _subjects(mask)
    if not figures or figures[0][0] < _MIN_MASK_COVERAGE * sw * sh:
        return crop_main_subject(src, dst, index)   # no clean figure -> whole body
    if index >= len(figures):
        index = 0
    _, x0, y0, x1, y1 = figures[index]
    # the head band: the top `band` of the figure, horizontally centred on the
    # foreground actually present in that band, so a tilted or off-axis head is
    # still centred rather than sliced by the figure's overall bounding box
    y_end = y0 + max(1, int((y1 - y0 + 1) * band))
    xs = [x for y in range(y0, min(y_end + 1, sh)) for x in range(x0, min(x1 + 1, sw))
          if mask[y][x]]
    if not xs:
        return crop_main_subject(src, dst, index)
    hx0, hx1 = min(xs), max(xs)
    cx, half = (hx0 + hx1) / 2, (hx1 - hx0 + 1) * 0.8   # 1.6x the head's width
    half = max(half, (y_end - y0 + 1) * 0.55)           # never narrower than tall
    f = max(w, h) / max(sw, sh)
    cx0, cx1 = max(0, int((cx - half) * f)), min(w, int((cx + half) * f))
    cy0, cy1 = max(0, int((y0 - (y_end - y0) * 0.12) * f)), min(h, int(y_end * f))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return crop_main_subject(src, dst, index)
    crop = img.crop((cx0, cy0, cx1, cy1))
    side = max(crop.size)
    pos = ((side - crop.size[0]) // 2, (side - crop.size[1]) // 2)
    try:
        from rembg import remove
        cut = remove(crop)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(cut, pos, cut)
    except Exception:
        canvas = Image.new("RGB", (side, side), _background_color(img.load(), w, h))
        canvas.paste(crop, pos)
    if side < upscale_to:   # a small crop carries no more detail, but the
        canvas = canvas.resize((upscale_to, upscale_to), Image.LANCZOS)  # sampler
    dst.parent.mkdir(parents=True, exist_ok=True)                        # gets room
    canvas.save(dst, "PNG")
    return dst


def crop_main_subject(src: Path, dst: Path, index: int = 0) -> Path:
    """Write a crop of one of src's figures to dst (PNG), largest figure first.

    `index` picks among the figures on a multi-hero sheet. It clamps to the
    largest rather than failing, so a plan asking for a second character on a
    single-character sheet still produces a mesh; the clamp is logged, because
    two characters silently built from one crop is the failure that matters.
    Falls back to the original image whenever analysis is inconclusive — never
    blocks a run.
    """
    try:
        img = Image.open(src).convert("RGB")
    except Exception:  # OSError, DecompressionBombError, ... — never block a run
        return src
    w, h = img.size
    scale = max(w, h) / _SCALE
    small = img.resize((max(1, round(w / scale)), max(1, round(h / scale)))) \
        if scale > 1 else img.copy()
    erode_px = _ERODE_PX   # breaks prop bridges; bbox is re-expanded below
    mask = _erode(_foreground_mask(small), erode_px)
    sw, sh = small.size
    figures = _subjects(mask)
    if not figures:
        return src  # nothing detected — don't guess
    # A matte that covers almost nothing has failed, it has not found a small
    # subject. rembg segments salient objects and people, so an interior or a
    # landscape returns near-empty: the library reference matted to 1.7% of the
    # frame and would have been cropped from 1672x941 down to a 323px fragment
    # and rebuilt as that fragment. When the mask is that thin the whole image
    # is the subject.
    if figures[0][0] < _MIN_MASK_COVERAGE * sw * sh:
        print(f"[refimage] {Path(src).name}: foreground is only "
              f"{figures[0][0] / (sw * sh):.1%} of the frame — using the whole image",
              flush=True)
        return src
    if index >= len(figures):
        print(f"[refimage] {Path(src).name}: asked for figure {index}, found "
              f"{len(figures)} — using the largest", flush=True)
        index = 0
    _, x0, y0, x1, y1 = figures[index]
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
    # square canvas, subject matted onto flat white: TRELLIS rebuilds whatever is
    # in frame, so a kept dark backdrop becomes a card mesh and stray sheet
    # furniture (measure lines, neighbours' props) becomes duplicate geometry
    side = max(crop.size)
    pos = ((side - crop.size[0]) // 2, (side - crop.size[1]) // 2)
    try:
        from rembg import remove
        cut = remove(crop)  # RGBA soft matte at crop resolution
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(cut, pos, cut)
    except Exception:  # rembg unavailable -> old behavior: sheet background color
        canvas = Image.new("RGB", (side, side), _background_color(img.load(), w, h))
        canvas.paste(crop, pos)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, "PNG")
    return dst
