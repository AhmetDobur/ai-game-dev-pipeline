"""Reference-image conditioning: crop, ref_image repair, scaffold environment."""
from pathlib import Path

from PIL import Image

from pipeline.decompose import repair_task_list, validate_task_list
from pipeline.refimage import crop_main_subject
from pipeline.scaffold import _world_tscn


def _sheet(path: Path) -> Path:
    """Synthetic character sheet: dark bg, one big figure + two small ones."""
    img = Image.new("RGB", (600, 400), (18, 16, 14))
    px = img.load()
    for x in range(80, 220):        # big figure (140x300)
        for y in range(50, 350):
            px[x, y] = (200, 180, 150)
    for x0 in (300, 450):           # two small figures (60x100)
        for x in range(x0, x0 + 60):
            for y in range(250, 350):
                px[x, y] = (190, 170, 140)
    img.save(path)
    return path


def test_crop_picks_largest_figure(tmp_path):
    src = _sheet(tmp_path / "sheet.png")
    out = crop_main_subject(src, tmp_path / "crop.png")
    assert out != src and out.exists()
    w, h = Image.open(out).size
    assert w == h  # square canvas
    # the big figure is 140x300 in a 600x400 sheet: crop must be far smaller
    # than the sheet but bigger than a small figure
    assert 200 < max(w, h) < 500


def test_crop_single_subject_passthrough(tmp_path):
    img = Image.new("RGB", (300, 300), (240, 240, 240))
    px = img.load()
    for x in range(10, 290):
        for y in range(10, 290):
            px[x, y] = (60, 60, 200)
    src = tmp_path / "solo.png"
    img.save(src)
    assert crop_main_subject(src, tmp_path / "c.png") == src


def _graph(ref=None):
    spec3d = {"prompt": "x"}
    if ref:
        spec3d["ref_image"] = ref
    return [
        {"id": "art", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "a knight, solo", "purpose": "concept"}},
        {"id": "mesh", "type": "design_3d", "depends_on": ["art"], "spec": spec3d},
        {"id": "code", "type": "code", "depends_on": [],
         "spec": {"file": "scripts/a.gd", "description": "x"}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]


def test_ref_image_fuzzy_repair():
    tasks = _graph(ref="char.png")   # router mangled the full path
    repair_task_list(tasks, ["inbox/started/123-x__char.png"])
    assert tasks[1]["spec"]["ref_image"] == "inbox/started/123-x__char.png"
    validate_task_list(tasks)


def test_ref_image_unknown_dropped_falls_back_to_concept():
    tasks = _graph(ref="nonsense.jpg")
    repair_task_list(tasks, ["inbox/started/123-x__char.png"])
    # unknown ref dropped; repair auto-links concept_from to the design_2d dep
    assert "ref_image" not in tasks[1]["spec"]
    assert tasks[1]["spec"]["concept_from"] == "art"
    validate_task_list(tasks)


def test_design_3d_with_only_ref_image_validates():
    tasks = _graph(ref="r.png")
    tasks[1]["depends_on"] = []      # no design_2d link at all
    repair_task_list(tasks, ["r.png"])
    validate_task_list(tasks)        # ref_image alone satisfies image conditioning


def test_world_tscn_environment_and_props():
    scene = _world_tscn("res://scripts/player.gd", "res://assets/c/character.glb",
                        ["res://assets/e/shelf.glb"])
    assert "volumetric_fog_enabled = true" in scene
    assert scene.count('type="OmniLight3D"') == 5           # candle ring
    # every prop instance is wrapped in a StaticBody3D with a collision shape:
    # 4 placements of the one env glb, plus the floor body
    assert scene.count('type="StaticBody3D"') == 5
    assert scene.count('shape = SubResource("prop_shape")') == 4
    assert scene.count("instance=ExtResource") == 5         # 4 props + player mesh
    assert "character.glb" in scene


def test_prop_concept_isolated_from_scene_ref():
    # a design_2d feeding TRELLIS loses its scene ref and gains isolation phrasing;
    # a design_2d feeding nothing keeps its ref (pure art / backdrop)
    tasks = [
        {"id": "shelf_art", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "a bookshelf", "ref_image": "lib.png"}},
        {"id": "shelf", "type": "design_3d", "depends_on": ["shelf_art"],
         "spec": {"prompt": "shelf mesh", "concept_from": "shelf_art"}},
        {"id": "backdrop", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "library mural", "ref_image": "lib.png"}},
        {"id": "player", "type": "code", "depends_on": [],
         "spec": {"file": "scripts/player.gd", "description": "controller"}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]
    repair_task_list(tasks, ["lib.png"])
    assert "ref_image" not in tasks[0]["spec"]
    assert "single isolated object" in tasks[0]["spec"]["prompt"]
    assert tasks[2]["spec"]["ref_image"] == "lib.png"
    assert "isolated" not in tasks[2]["spec"]["prompt"]
    validate_task_list(tasks)


def test_coerce_router_wrappers():
    from pipeline.decompose import _coerce_task_list
    tasks = [{"id": "a", "type": "code", "spec": {}}]
    assert _coerce_task_list({"tasks": tasks}) == tasks       # dict wrapper
    assert _coerce_task_list(tasks[0]) == tasks               # single bare task
    assert _coerce_task_list(tasks) == tasks                  # already a list
    assert _coerce_task_list({"a": 1}) == {"a": 1}            # not coercible


def test_think_block_stripped():
    from pipeline.decompose import _THINK_RE
    raw = '<think>maybe {"id": "x"} hmm</think>\n[{"id": "a"}]'
    assert _THINK_RE.sub("", raw).strip() == '[{"id": "a"}]'
    # unterminated think block (token budget ran out) strips to empty
    assert _THINK_RE.sub("", "<think>endless pondering").strip() == ""
