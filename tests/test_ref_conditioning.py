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
    # graph keyed by task id (id present or implied by the key)
    keyed = {"a": {"id": "a", "type": "code", "spec": {}},
             "b": {"type": "assemble", "spec": {}}}
    out = _coerce_task_list(keyed)
    assert out == [{"id": "a", "type": "code", "spec": {}},
                   {"id": "b", "type": "assemble", "spec": {}}]


def test_think_block_stripped():
    from pipeline.decompose import _THINK_RE
    raw = '<think>maybe {"id": "x"} hmm</think>\n[{"id": "a"}]'
    assert _THINK_RE.sub("", raw).strip() == '[{"id": "a"}]'
    # unterminated think block (token budget ran out) strips to empty
    assert _THINK_RE.sub("", "<think>endless pondering").strip() == ""


def test_router_omissions_repaired():
    # the graph run 755c000b5c86 actually got: no prop design_3ds at all, and
    # the user refs dropped. Repair must synthesize meshes and re-attach refs.
    tasks = [
        {"id": "char_art", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "an imam brawler, solo"}},
        {"id": "char_mesh", "type": "design_3d", "depends_on": ["char_art"],
         "spec": {"prompt": "mesh", "concept_from": "char_art"}},
        {"id": "anim", "type": "rig_animate", "depends_on": ["char_mesh"],
         "spec": {"mesh_from": "char_mesh", "body_plan": "humanoid",
                  "animations": ["idle"]}},
        {"id": "shelf", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "a massive bookshelf in a library hall"}},
        {"id": "env", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "a vast baroque library hall"}},
        {"id": "player", "type": "code", "depends_on": [],
         "spec": {"file": "scripts/p.gd", "description": "d"}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]
    refs = ["inbox/started/1-x__x_char.png", "inbox/started/1-x__x_env.png"]
    repair_task_list(tasks, refs)
    by = {t["id"]: t for t in tasks}
    assert by["shelf_mesh"]["spec"]["concept_from"] == "shelf"  # prop mesh synthesized
    assert "env_mesh" not in by                                 # scene art stays 2D
    assert by["char_mesh"]["spec"]["ref_image"] == refs[0]      # char ref enforced
    assert by["env"]["spec"]["ref_image"] == refs[1]            # env ref on backdrop
    assert "ref_image" not in by["shelf"]["spec"]               # isolation strips it
    assert "single isolated object" in by["shelf"]["spec"]["prompt"]
    validate_task_list(tasks)


def test_frame_data_only_on_combat_sim():
    fd = {"punch": {"startup": 3, "active": 2, "hitstun": 5,
                    "knockback": [1, 0], "tolerance": 0}}
    tasks = [
        {"id": "lib", "type": "code", "depends_on": [],
         "spec": {"file": "scripts/library.gd", "description": "env",
                  "frame_data": dict(fd)}},
        {"id": "sim", "type": "code", "depends_on": [],
         "spec": {"file": "scripts/combat_sim.gd", "description": "combat",
                  "frame_data": dict(fd)}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]
    repair_task_list(tasks)
    assert "frame_data" not in tasks[0]["spec"]   # typed junk on the wrong file
    assert tasks[1]["spec"]["frame_data"] == fd   # the real contract survives


def test_rig_on_art_retargeted_through_synthesized_mesh():
    # run c293365cffa0's router rigged the design_2d directly and put string
    # junk in frame_data; both must repair without a retry round
    tasks = [
        {"id": "char_art", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "an imam brawler, solo"}},
        {"id": "anim", "type": "rig_animate", "depends_on": ["char_art"],
         "spec": {"mesh_from": "char_art", "body_plan": "humanoid",
                  "animations": ["idle"]}},
        {"id": "player", "type": "code", "depends_on": [],
         "spec": {"file": "scripts/player.gd", "description": "d",
                  "frame_data": "frame_data"}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]
    repair_task_list(tasks, [])
    by = {t["id"]: t for t in tasks}
    assert by["anim"]["spec"]["mesh_from"] == "char_art_mesh"
    assert by["char_art_mesh"]["spec"]["concept_from"] == "char_art"
    assert "char_art_mesh" in by["anim"]["depends_on"]
    assert "frame_data" not in by["player"]["spec"]
    validate_task_list(tasks)


def test_style_tail_is_identical_across_sibling_props():
    """Sibling props must end with a byte-identical style tail. The seed and the
    negative prompt are already shared, so this is what makes them converge."""
    from pipeline.decompose import repair_task_list, _QUALITY_TAIL
    tasks = [
        {"id": "shelf_art", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "a bookshelf"}},
        {"id": "shelf", "type": "design_3d", "depends_on": ["shelf_art"],
         "spec": {"prompt": "bookshelf", "concept_from": "shelf_art"}},
        # a router that copied the few-shot already says "isolated" -- this task
        # used to skip the whole suffix append and diverge from its sibling
        {"id": "globe_art", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "a globe, single isolated object"}},
        {"id": "globe", "type": "design_3d", "depends_on": ["globe_art"],
         "spec": {"prompt": "globe", "concept_from": "globe_art"}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]
    repair_task_list(tasks, [], "Make a game.\nStyle: dark baroque oil painting\n")
    from pipeline.decompose import _ISO_CLAUSES
    for t in tasks:
        if t["type"] != "design_2d":
            continue
        p = t["spec"]["prompt"]
        assert p.endswith(f"dark baroque oil painting, {_QUALITY_TAIL}"), p
        assert "product render" not in p, p
        # every mesh-feeding concept carries all three isolation clauses, once
        for cl in _ISO_CLAUSES:
            assert p.lower().count(cl) == 1, (cl, p)


def test_quality_tail_applies_without_a_style_line():
    from pipeline.decompose import repair_task_list, _QUALITY_TAIL
    tasks = [{"id": "a", "type": "design_2d", "depends_on": [],
              "spec": {"prompt": "a crate"}},
             {"id": "build", "type": "assemble", "depends_on": [], "spec": {}}]
    repair_task_list(tasks, [], "no style line here")
    # a lone prop design_2d gets a synthesized mesh, hence the isolation clauses
    assert tasks[0]["spec"]["prompt"].endswith(_QUALITY_TAIL)
    assert tasks[0]["spec"]["prompt"].startswith("a crate,")


def test_char_ref_survives_a_prop_mesh_holding_its_own_ref():
    """A prop mesh with a ref_image used to starve every character mesh."""
    from pipeline.decompose import repair_task_list
    tasks = [
        {"id": "hero_mesh", "type": "design_3d", "depends_on": [],
         "spec": {"prompt": "the hero", "concept_from": "hero_art"}},
        {"id": "hero_art", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "the hero"}},
        {"id": "hero_anim", "type": "rig_animate", "depends_on": ["hero_mesh"],
         "spec": {"mesh_from": "hero_mesh", "body_plan": "humanoid",
                  "animations": ["idle"], "extras": []}},
        {"id": "prop", "type": "design_3d", "depends_on": [],
         "spec": {"prompt": "a lamp", "ref_image": "inbox/lamp_env.png"}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]
    repair_task_list(tasks, ["inbox/sheet_char.png", "inbox/lamp_env.png"], "")
    hero = next(t for t in tasks if t["id"] == "hero_mesh")
    assert hero["spec"]["ref_image"] == "inbox/sheet_char.png"
