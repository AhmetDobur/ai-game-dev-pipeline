"""Reference-image conditioning: crop, ref_image repair, scaffold environment."""
import pytest
import json
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


def test_world_tscn_builds_an_enclosed_hall():
    """The room is authored geometry: TRELLIS cannot make a space you stand in,
    so a walkable interior has to exist before any generated prop is placed."""
    scene = _world_tscn("res://scripts/player.gd", ["res://assets/c/character.glb"],
                        ["res://assets/e/shelf.glb"])
    assert "volumetric_fog_enabled = true" in scene
    # four walls, a ceiling and a floor -- without these the player walked on an
    # unbounded plane, which is what "a white floor and nothing else" was
    for part in ("WallL", "WallR", "WallBack", "WallFront", "Ceiling", "Floor"):
        assert f'name="{part}"' in scene, part
    assert scene.count('type="CylinderMesh"') == 12         # colonnade, both sides
    assert 'name="RoseWindow"' in scene and "emission_enabled = true" in scene
    assert 'name="Apse"' in scene and 'name="Step2"' in scene
    assert "character.glb" in scene
    # no unbounded ground plane left behind
    assert "400, 1, 400" not in scene


def test_world_tscn_props_line_the_walls_not_a_ring():
    scene = _world_tscn(None, [], ["res://assets/e/shelf.glb"])
    assert scene.count("instance=ExtResource") == 3         # 3 placements, no player mesh
    assert scene.count('shape = SubResource("prop_shape")') == 3
    from pipeline.scaffold import _wall_props, HALL_W
    for _, x, _z, _yaw in _wall_props(["a"]):
        assert abs(abs(x) - (HALL_W / 2 - 4.2)) < 0.01      # against a side wall


def test_hall_palette_follows_the_environment_reference(tmp_path):
    """Room materials are read off the run's reference, not hardcoded."""
    from PIL import Image
    ref = tmp_path / "lib_env.png"
    Image.new("RGB", (64, 64), (20, 90, 160)).save(ref)     # strongly blue
    scene = _world_tscn(None, [], [], str(ref))
    assert "Color(0.078, 0.353, 0.627" in scene, [
        l for l in scene.splitlines() if "albedo_color" in l]


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
    assert "isolated on a plain flat white background" in tasks[0]["spec"]["prompt"]
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
    assert "isolated on a plain flat white background" in by["shelf"]["spec"]["prompt"]
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
    from pipeline.decompose import _ISO_COMMON
    for t in tasks:
        if t["type"] != "design_2d":
            continue
        p = t["spec"]["prompt"]
        # mesh-feeding concepts get the TREATMENT tail only -- the operator's
        # scene-naming style line would put them back inside a room
        assert p.endswith(_QUALITY_TAIL), p
        assert "dark baroque oil painting" not in p, p
        assert "product render" not in p, p
        for cl in _ISO_COMMON:
            assert p.lower().count(cl) == 1, (cl, p)


def test_backdrop_art_keeps_the_operator_style_line():
    """Art no mesh is built from is the one place the scene style belongs."""
    from pipeline.decompose import repair_task_list, _QUALITY_TAIL
    tasks = [{"id": "sky", "type": "design_2d", "depends_on": [],
              "spec": {"prompt": "a vast library hall backdrop"}},
             {"id": "build", "type": "assemble", "depends_on": [], "spec": {}}]
    repair_task_list(tasks, [], "Style: dark baroque oil painting\n")
    p = tasks[0]["spec"]["prompt"]
    assert p.endswith(f"dark baroque oil painting, {_QUALITY_TAIL}"), p
    assert "studio cutout" not in p, p


def test_prop_is_not_described_as_standing_head_to_toe():
    """Character framing on a bookshelf produced nonsense concept art."""
    from pipeline.decompose import repair_task_list
    tasks = [{"id": "shelf_art", "type": "design_2d", "depends_on": [],
              "spec": {"prompt": "a bookshelf"}},
             {"id": "shelf", "type": "design_3d", "depends_on": ["shelf_art"],
              "spec": {"prompt": "bookshelf", "concept_from": "shelf_art"}},
             {"id": "build", "type": "assemble", "depends_on": [], "spec": {}}]
    repair_task_list(tasks, [], "")
    p = tasks[0]["spec"]["prompt"].lower()
    assert "single isolated object" in p, p
    for bad in ("head to toe", "standing", "full body"):
        assert bad not in p, (bad, p)


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


# --- geometry gate -------------------------------------------------------
# Measured bboxes from 21 real pipeline meshes. Numbers 2/3/4/21 came back as
# flat planes -- 3 mm thick, 8 MB, 40k faces -- and every byte-count check
# passed them. 14 is the mesh the user called his ideal.
_REAL_BBOXES = {
    1: [0.9969, 0.2824, 0.9705], 2: [1.0019, 0.003, 0.9763],
    3: [0.996, 0.0031, 0.9954], 4: [0.9957, 0.0029, 0.9955],
    5: [0.9965, 0.9959, 0.7306], 13: [0.8848, 0.5139, 0.9918],
    14: [0.5241, 0.3349, 0.9984], 15: [0.9749, 0.3509, 0.7616],
    20: [0.9976, 0.1287, 0.9857], 21: [0.9929, 0.0129, 1.0011],
}
_BROKEN = {2, 3, 4, 21}


def _write_metrics(tmp_path, bbox):
    mesh = tmp_path / "m.glb"
    mesh.write_bytes(b"x" * 10)
    (tmp_path / "m.glb.metrics.json").write_text(json.dumps({"bbox": bbox}))
    return mesh


def test_geometry_gate_matches_the_measured_meshes(tmp_path):
    from pipeline.inspect3d import verdict
    for n, bbox in _REAL_BBOXES.items():
        d = tmp_path / str(n)
        d.mkdir()
        ok, detail = verdict(_write_metrics(d, bbox))
        assert ok is (n not in _BROKEN), (n, bbox, detail)


def test_missing_metrics_never_rejects(tmp_path):
    """A preview that did not run must not fail the task."""
    from pipeline.inspect3d import verdict
    mesh = tmp_path / "m.glb"
    mesh.write_bytes(b"x" * 10)
    assert verdict(mesh)[0] is True


def test_validate_rejects_a_flat_plane(tmp_path):
    from pipeline.validate import validate
    mesh = _write_metrics(tmp_path, [1.0019, 0.003, 0.9763])
    mesh.write_bytes(b"x" * 60_000)      # comfortably over MIN_MESH_BYTES
    ok, detail = validate({"type": "design_3d", "spec": {}}, [mesh])
    assert ok is False and "degenerate" in detail


def test_assemble_validation_ignores_the_world_screenshot(tmp_path):
    """The build is the artifact under test; the screenshot rides along."""
    from pipeline.validate import validate
    exe = tmp_path / "game.exe"
    exe.write_bytes(b"x" * 2_000_000)
    shot = tmp_path / "world_shot.png"
    shot.write_bytes(b"x" * 30_000)
    ok, detail = validate({"type": "assemble", "spec": {}}, [exe, shot])
    assert ok is True, detail


def _duo(path: Path) -> Path:
    """Two heroes of comparable size plus a squat prop — a two-character sheet."""
    img = Image.new("RGB", (600, 400), (18, 16, 14))
    px = img.load()
    for x in range(60, 200):        # hero A, 140x320
        for y in range(40, 360):
            px[x, y] = (200, 180, 150)
    for x in range(320, 430):       # hero B, 110x290
        for y in range(60, 350):
            px[x, y] = (190, 170, 140)
    for x in range(470, 510):       # prop: small and wider-than-tall-ish
        for y in range(300, 350):
            px[x, y] = (180, 160, 130)
    img.save(path)
    return path


def test_crop_index_selects_second_hero(tmp_path):
    """Both characters must be reachable: handing every hero the same crop is
    how the pipeline shipped one character twice."""
    src = _duo(tmp_path / "duo.png")
    a = crop_main_subject(src, tmp_path / "a.png", 0)
    b = crop_main_subject(src, tmp_path / "b.png", 1)
    assert a.exists() and b.exists()
    assert a.read_bytes() != b.read_bytes()
    assert Image.open(a).size[0] > Image.open(b).size[0]   # A is the larger hero


def test_crop_index_ignores_props(tmp_path):
    """The prop is neither big enough nor tall enough to count as a figure, so
    asking for a third subject clamps back to the largest."""
    src = _duo(tmp_path / "duo.png")
    a = crop_main_subject(src, tmp_path / "a.png", 0)
    c = crop_main_subject(src, tmp_path / "c.png", 2)
    assert a.read_bytes() == c.read_bytes()


def test_two_char_meshes_get_different_subjects():
    """decompose must hand the two character meshes different figures."""
    tasks = [
        {"id": "m1", "type": "design_3d", "deps": [], "spec": {"prompt": "hero one"}},
        {"id": "m2", "type": "design_3d", "deps": [], "spec": {"prompt": "hero two"}},
        {"id": "r1", "type": "rig_animate", "deps": ["m1"], "spec": {"mesh_from": "m1"}},
        {"id": "r2", "type": "rig_animate", "deps": ["m2"], "spec": {"mesh_from": "m2"}},
    ]
    repair_task_list(tasks, ["refs/duo_char.png"])
    meshes = {t["id"]: t["spec"] for t in tasks if t.get("type") == "design_3d"}
    assert meshes["m1"]["ref_image"] == meshes["m2"]["ref_image"]  # one sheet
    subjects = {meshes["m1"].get("ref_subject", 0), meshes["m2"].get("ref_subject", 0)}
    assert subjects == {0, 1}, f"both meshes took the same figure: {subjects}"


def test_two_characters_get_split_screen():
    """Both heroes on screen at once: one body each, one camera each, and the
    cameras inside SubViewports so each renders its own half."""
    from pipeline.scaffold import _world_tscn
    scene = _world_tscn("res://scripts/player.gd",
                        ["res://assets/a/character.glb",
                         "res://assets/b/character.glb"], [])
    assert scene.count('type="CharacterBody3D"') == 2
    assert scene.count('type="SubViewport"') == 2
    assert scene.count('type="Camera3D"') == 2
    assert "device = 0" in scene and "device = 1" in scene
    assert 'camera_path = NodePath("../Split/Half1/View/Camera")' in scene
    # each player instances a DIFFERENT mesh
    assert "assets/a/character.glb" in scene and "assets/b/character.glb" in scene


def test_one_character_keeps_single_camera():
    """A solo run must not grow split-screen machinery."""
    from pipeline.scaffold import _world_tscn
    scene = _world_tscn("res://scripts/player.gd", ["res://assets/a/character.glb"], [])
    assert "SubViewport" not in scene
    assert scene.count('type="Camera3D"') == 1
    assert '[node name="Player" type="CharacterBody3D"' in scene


def test_players_face_the_apse():
    """Spawns sit at the hall's near end; unrotated bodies face -Z and would
    start nose-to-wall with the whole room behind them."""
    from pipeline.scaffold import _world_tscn, SPAWN_Z
    scene = _world_tscn("res://scripts/player.gd", ["res://a/character.glb"], [])
    assert f"Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, 0.0, 1.2, {SPAWN_Z})" in scene
    assert "position = Vector3(0.0, 1.2," not in scene   # no unrotated spawn left


def test_character_mesh_is_turned_to_face_forward():
    """A TRELLIS mesh looks down its own +Z; a Godot body walks along -Z. Without
    the flip the character moonwalks, facing its own camera."""
    from pipeline.scaffold import _world_tscn
    scene = _world_tscn(None, ["res://a/character.glb"], [])
    mesh = scene.split('[node name="Mesh"')[1]
    assert "Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0)" in mesh.split("[node")[0]


def _glb(path: Path, doc: dict) -> Path:
    """Minimal GLB container around a glTF JSON document."""
    import struct
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I", len(js)) + b"JSON" + js
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    return path


def test_rig_without_skin_is_rejected(tmp_path):
    """The pipeline's quietest failure: right size, every named clip, imports
    clean -- and the bones animate nothing."""
    from pipeline.inspect3d import has_skin, rig_verdict
    dead = _glb(tmp_path / "dead.glb", {
        "asset": {"version": "2.0"},
        "nodes": [{"mesh": 0, "name": "Mesh_0"}, {"name": "pelvis"}],
        "meshes": [{"primitives": [{"attributes": {"JOINTS_0": 0, "WEIGHTS_0": 1}}]}],
        "animations": [{"name": "run", "channels": [], "samplers": []}],
    })
    assert has_skin(dead) is False
    ok, why = rig_verdict(dead)
    assert not ok and "skin" in why


def test_rig_with_skin_passes(tmp_path):
    from pipeline.inspect3d import has_skin, rig_verdict
    live = _glb(tmp_path / "live.glb", {
        "asset": {"version": "2.0"},
        "nodes": [{"mesh": 0, "name": "Mesh_0", "skin": 0}, {"name": "pelvis"}],
        "meshes": [{"primitives": [{"attributes": {"JOINTS_0": 0, "WEIGHTS_0": 1}}]}],
        "skins": [{"joints": [1], "inverseBindMatrices": 2}],
    })
    assert has_skin(live) is True
    assert rig_verdict(live)[0]


def test_unreadable_model_is_never_rejected(tmp_path):
    """Same rule as the geometry gate: only an objectively broken signal fails a
    task, because a false rejection costs a full regeneration."""
    from pipeline.inspect3d import has_skin, rig_verdict
    junk = tmp_path / "junk.glb"
    junk.write_bytes(b"not a glb at all")
    assert has_skin(junk) is None
    assert rig_verdict(junk)[0]


def test_keyboard_seat_can_walk_not_only_sprint():
    """Input.get_vector always returns magnitude 1.0 for a key press, so a
    stick-throw run test applied to keyboard input made the seat sprint
    permanently and put the walk animation out of reach."""
    gd = Path("templates/player.gd").read_text()
    body = gd.split("func _physics_process")[1]
    # the run test is computed from the STICK only, before the keyboard fallback
    assert "var running := stick.length() > RUN_STICK" in body
    assert "running = Input.is_key_pressed(KEY_SHIFT if device == 0 else KEY_CTRL)" in body
    assert "move.length()" not in body      # never re-derived from merged input


def test_second_seat_has_its_own_keyboard_bindings():
    """Split-screen must be playable on one keyboard: the arrows used to be
    aliases of WASD, so they drove seat 1 and seat 2 had no binding at all."""
    from pipeline.scaffold import _KEYS
    for a in ("move_left", "move_right", "move_forward", "move_back"):
        assert a in _KEYS and f"p2_{a}" in _KEYS
        assert _KEYS[a] != _KEYS[f"p2_{a}"]
    for a in ("p2_look_left", "p2_look_right", "p2_look_up", "p2_look_down"):
        assert a in _KEYS
    assert len(set(_KEYS.values())) == len(_KEYS)   # no key drives two actions


def test_camera_hangs_off_a_spring_arm():
    """A fixed 3.5 m boom buries the camera under the floor slab past ~35 deg of
    pitch and sweeps it through walls and columns."""
    from pipeline.scaffold import _world_tscn
    for chars in (["res://a/character.glb"], ["res://a/c.glb", "res://b/c.glb"]):
        scene = _world_tscn("res://scripts/player.gd", chars, [])
        assert scene.count('type="SpringArm3D"') == len(chars)
        assert "spring_length = 3.5" in scene
        assert 'parent="Player/CamPivot"' in scene or 'parent="Player1/CamPivot"' in scene


def test_apse_stair_is_climbable_and_not_inside_the_platform():
    """Two of the three steps used to sit inside the Apse block, and a 0.35 m
    riser is a wall to a CharacterBody3D."""
    import re
    from pipeline.scaffold import _world_tscn, HALL_L
    scene = _world_tscn(None, [], [])
    zs = {}
    for name in ("Step0", "Step1", "Step2", "Apse"):
        m = re.search(rf'\[node name="{name}" type="StaticBody3D"[^\n]*\]\n'
                      r"transform = Transform3D\([^)]*?([-\d.]+)\)", scene)
        assert m, name
        zs[name] = float(m.group(1))
    apse_front = zs["Apse"] - 2.0
    assert zs["Step2"] + 0.6 <= apse_front + 1e-6, "top step is inside the apse"
    assert zs["Step0"] < zs["Step1"] < zs["Step2"], "steps out of order"
    assert "StairRamp" in scene, "no walkable ramp over the risers"


def _manifest_rows():
    return [
        {"id": "r-hero_art", "type": "design_2d", "summary": "a knight, solo",
         "observed": "1024x1024 image"},
        {"id": "r-hero_mesh", "type": "design_3d", "summary": "the knight mesh",
         "observed": "size 0.9x1.8x0.6; 40213 faces"},
        {"id": "r-hero_anim", "type": "rig_animate", "summary": "idle,walk,run",
         "observed": "idle (procedural, 1.0s, 7 bones), walk (procedural, 1.0s, 7 bones)"},
        {"id": "r-player", "type": "code", "summary": "scripts/player.gd",
         "observed": "48 lines"},
    ]


def test_animation_instruction_must_touch_the_rig():
    """The main mistake this guards: asked about the animation, the router goes
    and rewrites the concept art instead."""
    from pipeline.patch import check_patch_grounding
    rows = _manifest_rows()
    wrong = [{"target": "r-hero_art", "spec": {"prompt": "a more realistic knight"}}]
    try:
        check_patch_grounding(wrong, rows, "make the animation more realistic")
        raise AssertionError("a patch that ignores the rig was accepted")
    except ValueError as e:
        # the message must hand the router the facts it needs to correct itself
        assert "r-hero_anim" in str(e) and "procedural" in str(e)

    right = [{"target": "r-hero_anim",
              "spec": {"mesh_from": "r-hero_mesh", "body_plan": "humanoid",
                       "animations": ["idle", "walk", "run"], "extras": []}}]
    check_patch_grounding(right, rows, "make the animation more realistic")


def test_grounding_allows_an_add_of_the_right_type():
    from pipeline.patch import check_patch_grounding
    add = [{"id": "new_anim", "type": "rig_animate", "depends_on": ["r-hero_mesh"],
            "spec": {"mesh_from": "r-hero_mesh", "animations": ["attack"]}}]
    check_patch_grounding(add, _manifest_rows(), "add an attack animation")


def test_grounding_stays_quiet_when_the_instruction_is_ambiguous():
    """Only unambiguous misses may fail a patch -- a false rejection burns three
    router round-trips and then the whole revision."""
    from pipeline.patch import check_patch_grounding, instruction_domains
    rows = _manifest_rows()
    # two domains named at once -> no opinion
    assert len(instruction_domains("improve the mesh and the animation")) == 2
    check_patch_grounding([{"target": "r-player", "spec": {"file": "scripts/player.gd",
                                                           "description": "x"}}],
                          rows, "improve the mesh and the animation")
    # no domain named at all -> no opinion
    check_patch_grounding([{"target": "r-player", "spec": {"file": "scripts/player.gd",
                                                           "description": "x"}}],
                          rows, "make the game better")
    # domain named but the game has none of it -> no opinion
    check_patch_grounding([{"target": "r-player", "spec": {"file": "scripts/player.gd",
                                                           "description": "x"}}],
                          [r for r in rows if r["type"] != "audio"],
                          "add a voice line")


def test_manifest_carries_observed_facts(tmp_path):
    """The router must be shown what an artifact IS, not only what it was asked
    to be -- the two diverge constantly."""
    from pipeline.patch import manifest
    glb = _glb(tmp_path / "character.glb", {
        "asset": {"version": "2.0"},
        "nodes": [{"mesh": 0, "name": "Mesh_0"}, {"name": "pelvis"}],
        "meshes": [{"primitives": [{"attributes": {"JOINTS_0": 0}}]}],
        "accessors": [{"max": [1.5]}],
        "animations": [{"name": "run", "channels": [{"target": {"node": 1, "path": "rotation"}}],
                        "samplers": [{"input": 0}]}],
    })
    rows = manifest([{"id": "a-anim", "type": "rig_animate",
                      "spec": {"animations": ["idle", "walk", "run"]},
                      "output_path": str(glb)}])
    obs = rows[0]["observed"]
    assert "run" in obs and "1.5s" in obs
    assert "NOT SKINNED" in obs          # the spec would never have said so
    assert rows[0]["summary"] == "idle,walk,run"   # what was asked for, unchanged


def test_mesh_facts_come_from_the_glb_when_no_preview_ran(tmp_path):
    """The preview stage rarely runs, and a mesh sitting on disk should never be
    reported to the router as 'not measured'."""
    import struct
    from pipeline.inspect3d import geometry
    from pipeline.observe import facts_for
    doc = {
        "asset": {"version": "2.0"},
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [{"min": [-0.45, 0.0, -0.3], "max": [0.45, 1.8, 0.3], "count": 8},
                      {"count": 120600}],
    }
    mesh = _glb(tmp_path / "hero.glb", doc)
    g = geometry(mesh)
    assert g["dims"] == [0.9, 1.8, 0.6] and g["faces"] == 40200
    obs = facts_for("design_3d", str(mesh))
    assert "0.9x1.8x0.6" in obs and "40200 faces" in obs


def test_flat_plane_is_reported_from_the_glb(tmp_path):
    """TRELLIS returns 3 mm sheets that pass every byte check; the router must be
    told, not left to infer it from a prompt that still says 'a bookshelf'."""
    from pipeline.observe import facts_for
    doc = {
        "asset": {"version": "2.0"},
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"min": [-0.5, -0.0015, -0.5], "max": [0.5, 0.0015, 0.5],
                       "count": 300}],
    }
    assert "FLAT PLANE" in facts_for("design_3d", str(_glb(tmp_path / "flat.glb", doc)))


def _rows_with_specs():
    return [
        {"id": "349edb5bf375-hero_art", "type": "design_2d", "summary": "a knight",
         "observed": "1024x1024 image", "spec": {"prompt": "a knight, solo"}},
        {"id": "349edb5bf375-hero_mesh", "type": "design_3d", "summary": "knight mesh",
         "observed": "size 0.5x1.0x0.3",
         "spec": {"prompt": "knight mesh", "concept_from": "349edb5bf375-hero_art"}},
        {"id": "349edb5bf375-hero_anim", "type": "rig_animate", "summary": "idle,walk,run",
         "observed": "NOT SKINNED: the mesh is not bound to the skeleton",
         "spec": {"mesh_from": "349edb5bf375-hero_mesh", "body_plan": "humanoid",
                  "animations": ["idle", "walk", "run"], "extras": []}},
    ]


def test_bare_ids_are_repaired_not_rejected():
    """R1's real reply used 'hero_mesh'; the manifest ids are run-prefixed. The
    patch was semantically right and was thrown away three times over the name."""
    from pipeline.patch import repair_patch_list
    ops = [{"id": "hero_skinning", "type": "rig_animate",
            "depends_on": ["hero_mesh"], "spec": {"mesh_from": "hero_mesh"}}]
    repair_patch_list(ops, _rows_with_specs())
    assert ops[0]["depends_on"] == ["349edb5bf375-hero_mesh"]
    assert ops[0]["spec"]["mesh_from"] == "349edb5bf375-hero_mesh"


def test_ambiguous_bare_id_is_left_for_the_validator():
    from pipeline.patch import repair_patch_list
    rows = [{"id": "a-mesh", "type": "design_3d", "summary": "", "observed": "", "spec": {}},
            {"id": "b-mesh", "type": "design_3d", "summary": "", "observed": "", "spec": {}}]
    ops = [{"target": "mesh", "spec": {}}]
    repair_patch_list(ops, rows)
    assert ops[0]["target"] == "mesh"      # untouched: two candidates


def test_duplicate_add_becomes_a_modify():
    """Told the rig is unskinned, R1 added a SECOND rig for the same mesh —
    which would ship the character twice."""
    from pipeline.patch import collapse_duplicate_adds
    rows = _rows_with_specs()
    specs = {r["id"]: r["spec"] for r in rows}
    ops = [{"id": "hero_skinning", "type": "rig_animate",
            "depends_on": ["349edb5bf375-hero_mesh"],
            "spec": {"mesh_from": "349edb5bf375-hero_mesh", "animations": ["idle", "walk", "run"]}}]
    collapse_duplicate_adds(ops, rows, specs)
    assert ops[0] == {"target": "349edb5bf375-hero_anim",
                      "spec": {"mesh_from": "349edb5bf375-hero_mesh", "body_plan": "humanoid",
                               "animations": ["idle", "walk", "run"], "extras": []}}


def test_invented_spec_keys_are_reported_with_the_real_shape():
    """R1 invented {"skin_to_rig": true}; the message must say what a rig spec
    actually needs, because that message is what it gets to retry from."""
    from pipeline.patch import validate_patch_specs
    try:
        validate_patch_specs([{"id": "x", "type": "rig_animate",
                               "spec": {"skin_to_rig": True}}])
        raise AssertionError("an invented spec was accepted")
    except ValueError as e:
        for k in ("mesh_from", "body_plan", "animations"):
            assert k in str(e)


def test_router_sees_short_ids_and_either_form_is_accepted():
    """A 7B will not copy twenty 25-character run-scoped ids exactly — in a live
    run it wrote the bare form regardless. Show the short form, accept both."""
    from pipeline.patch import repair_patch_list, short_id
    assert short_id("349edb5bf375-hero_anim") == "hero_anim"
    assert short_id("hero_anim") == "hero_anim"          # already short
    assert short_id("notarunid1234-x") == "notarunid1234-x"   # not 12 hex
    rows = _rows_with_specs()
    for written in ("hero_mesh", "349edb5bf375-hero_mesh"):
        ops = [{"target": written, "spec": {}}]
        repair_patch_list(ops, rows)
        assert ops[0]["target"] == "349edb5bf375-hero_mesh", written


def test_single_candidate_grounding_hands_over_the_exact_reply():
    """The check exists to be grounded in measured facts, not to make a 7B
    rediscover the target on its third try — which it demonstrably did not."""
    import json as _json
    from pipeline.patch import check_patch_grounding
    rows = _rows_with_specs()
    try:
        check_patch_grounding([{"target": "349edb5bf375-hero_art", "spec": {"prompt": "x"}}],
                              rows, "make the animation more realistic")
        raise AssertionError("wrong-target patch accepted")
    except ValueError as e:
        msg = str(e)
        assert '"target": "hero_anim"' in msg          # short id, ready to copy
        assert "NOT SKINNED" in msg                    # the measured reason
        # the handed-over spec must be the real current one, valid to send back
        payload = _json.loads(msg[msg.index("[{"):msg.index("}]") + 2])
        assert payload[0]["spec"]["mesh_from"] == "349edb5bf375-hero_mesh"
        assert payload[0]["spec"]["body_plan"] == "humanoid"


# --- patch grounding + spec repair: five defects an adversarial review found ---

_MANIFEST = [
    {"id": "r1-hero_art", "type": "design_2d", "observed": "", "spec": {}},
    {"id": "r1-hero_mesh", "type": "design_3d", "observed": "", "spec": {}},
    {"id": "r1-hero_anim", "type": "rig_animate", "observed": "", "spec": {}},
]
_PARENT_SPECS = {
    "r1-hero_art": {"prompt": "a knight"},
    "r1-hero_mesh": {"prompt": "a knight", "concept_from": "r1-hero_art"},
    "r1-hero_anim": {"mesh_from": "r1-hero_mesh", "body_plan": "humanoid",
                     "animations": ["idle", "walk"], "extras": []},
}


def test_domain_words_match_whole_words_only():
    """'trigger'/'upright'/'walkway' are not animation instructions."""
    from pipeline.patch import instruction_domains
    for text in ("make the attack trigger sooner", "keep the statue upright",
                 "restore the original hitbox timing", "add a walkway to the apse"):
        assert instruction_domains(text) == set(), text
    # ...and the canonical phrasing still resolves
    assert instruction_domains("the character does not move") == {"rig_animate"}
    assert instruction_domains("make the player run faster") == {"rig_animate"}


def test_modify_spec_merges_over_parent_instead_of_replacing():
    """A partial MODIFY must not drop the keys the executor indexes."""
    from pipeline.patch import build_patch_graph
    parent = [{"id": "r1-hero_anim", "type": "rig_animate", "status": "done",
               "spec": dict(_PARENT_SPECS["r1-hero_anim"]), "depends_on": [],
               "attempts": 0, "output_path": "x.glb", "error": ""}]
    rows, _ = build_patch_graph(parent, [{"target": "r1-hero_anim",
                                          "spec": {"animations": ["idle", "walk", "run"]}}],
                                "r1", "r2")
    spec = next(r["spec"] for r in rows if r["type"] == "rig_animate")
    assert spec["animations"] == ["idle", "walk", "run"]
    assert spec["mesh_from"] == "r2-hero_mesh", "mesh_from was wiped by the MODIFY"
    assert spec["body_plan"] == "humanoid"


def test_add_op_gains_dependency_implied_by_its_spec():
    from pipeline.patch import repair_patch_list
    ops = [{"id": "attack_anim", "type": "rig_animate", "depends_on": [],
            "spec": {"mesh_from": "hero_mesh", "body_plan": "humanoid",
                     "animations": ["attack"]}}]
    repair_patch_list(ops, _MANIFEST)
    assert ops[0]["depends_on"] == ["r1-hero_mesh"]


def test_a_second_artifact_from_the_same_source_is_not_collapsed():
    from pipeline.patch import collapse_duplicate_adds
    ops = [{"id": "statue2", "type": "design_3d", "depends_on": ["r1-hero_art"],
            "spec": {"prompt": "a second knight statue", "concept_from": "r1-hero_art"}}]
    collapse_duplicate_adds(ops, _MANIFEST, _PARENT_SPECS)
    assert "target" not in ops[0], "a differently-described ADD was eaten as a MODIFY"
    # a genuine re-emission of the same thing still collapses
    dup = [{"id": "hero_mesh_again", "type": "design_3d", "depends_on": [],
            "spec": {"prompt": "a knight", "concept_from": "r1-hero_art"}}]
    collapse_duplicate_adds(dup, _MANIFEST, _PARENT_SPECS)
    assert dup[0].get("target") == "r1-hero_mesh"


def test_modify_specs_are_validated_against_the_targets_real_type():
    import pytest
    from pipeline.patch import validate_patch_specs
    # merged spec is complete -> accepted
    validate_patch_specs([{"target": "r1-hero_anim", "spec": {"animations": ["run"]}}],
                         _MANIFEST, _PARENT_SPECS)
    # parent had nothing to merge -> the missing keys are reported, not skipped
    with pytest.raises(ValueError, match="mesh_from"):
        validate_patch_specs([{"target": "r1-hero_anim", "spec": {"skin_to_rig": True}}],
                             _MANIFEST, {})


def test_extract_json_picks_one_value_when_the_reply_holds_two():
    """R1 restating its plan produced '[...][...]', which span extraction --
    first bracket to last bracket -- cannot parse, and it killed a planning run.
    """
    from pipeline.adapters.llm import extract_json
    two = ('[{"id": "a", "type": "design_2d"}]\n\n'
           '[{"id": "b", "type": "design_3d"}, {"id": "c", "type": "assemble"}]')
    assert extract_json(two) == [{"id": "b", "type": "design_3d"},
                                 {"id": "c", "type": "assemble"}]
    assert extract_json('[{"id": "z"}] and that completes the plan.') == [{"id": "z"}]
    assert extract_json('```json\n[{"id": "x"}]\n```') == [{"id": "x"}]


def test_unknown_dependency_is_resolved_or_dropped_not_fatal():
    """A near-miss dependency id must not reject an otherwise valid plan."""
    from pipeline.decompose import repair_task_list
    tasks = [
        {"id": "char_0_art", "type": "design_2d", "depends_on": [],
         "spec": {"prompt": "a knight"}},
        {"id": "char_0_mesh", "type": "design_3d", "depends_on": ["char_0_art"],
         "spec": {"prompt": "a knight", "concept_from": "char_0_art"}},
        # the router's own typo, and an id that resolves to nothing at all
        {"id": "char_0_anim", "type": "rig_animate",
         "depends_on": ["character_0_mesh", "totally_unrelated_xyz"],
         "spec": {"mesh_from": "char_0_mesh", "body_plan": "humanoid",
                  "animations": ["idle"]}},
    ]
    repair_task_list(tasks)
    deps = next(t for t in tasks if t["id"] == "char_0_anim")["depends_on"]
    assert "char_0_mesh" in deps
    assert "character_0_mesh" not in deps and "totally_unrelated_xyz" not in deps


def test_a_near_empty_matte_falls_back_to_the_whole_image(tmp_path, monkeypatch):
    """rembg segments salient objects; an interior returns almost nothing.

    Cropping to that near-empty matte turned a 1672x941 library photo into a
    323px fragment, and the mesh would have been built from the fragment.
    """
    from PIL import Image
    from pipeline import refimage

    src = tmp_path / "interior.png"
    Image.new("RGB", (400, 200), (90, 90, 100)).save(src)
    # a matte that finds a 10x10 speck: 0.25% of the frame, far below the floor
    monkeypatch.setattr(refimage, "_foreground_mask",
                        lambda img: [[20 <= x < 30 and 20 <= y < 30
                                      for x in range(img.size[0])]
                                     for y in range(img.size[1])])
    out = refimage.crop_main_subject(src, tmp_path / "out.png", 0)
    assert Path(out).resolve() == src.resolve(), "a failed matte must not crop"


def _figure_png(path, w=400, h=900):
    """A crude standing figure on white: small head blob over a wide body."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([170, 40, 230, 130], fill=(20, 20, 20))          # head
    d.rectangle([120, 130, 280, 820], fill=(20, 20, 20))       # body
    img.save(path)
    return path


def test_crop_head_frames_the_head_not_the_body(tmp_path):
    from pipeline.refimage import crop_head
    src = _figure_png(tmp_path / "fig.png")
    out = crop_head(src, tmp_path / "head.png")
    from PIL import Image
    im = Image.open(out)
    assert im.size[0] == im.size[1]          # square canvas for the sampler
    assert im.size[0] >= 1024                # upscaled: a head crop is small
    # the crop must come from the top of the figure, not the middle of the torso
    dark = sum(1 for px in list(im.convert("L").getdata()) if px < 128)
    assert dark > 0, "cropped an empty region"


def test_crop_head_falls_back_when_no_figure(tmp_path):
    from pipeline.refimage import crop_head
    from PIL import Image
    blank = tmp_path / "blank.png"
    Image.new("RGB", (300, 300), (255, 255, 255)).save(blank)
    # nothing detectable -> returns the source rather than a garbage fragment
    assert crop_head(blank, tmp_path / "h.png") == blank


def test_head_mesh_is_exempt_from_the_standing_figure_check():
    from pipeline.validate import _is_humanoid
    body = {"id": "hero_mesh", "spec": {"prompt": "the character"}}
    head = {"id": "hero_mesh_head", "spec": {"prompt": "the character", "detail": "head"}}
    assert _is_humanoid(body) and not _is_humanoid(head)


def test_comfy_wait_resets_its_clock_while_the_server_makes_progress(monkeypatch, tmp_path):
    """A total wall clock killed a still-running 10h job at 4h and re-queued it,
    leaving a duplicate behind the copy that was about to finish. Only a prompt
    that has genuinely stopped moving may time out."""
    import pipeline.adapters.comfy as comfy_mod

    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "X", "inputs": {}}}))
    clock = {"t": 0.0}
    monkeypatch.setattr(comfy_mod.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 60))
    monkeypatch.setattr(comfy_mod.time, "time", lambda: clock["t"])

    polls = {"n": 0}

    class R:
        def __init__(self, payload=None, text=""):
            self.payload, self.text, self.ok = payload, text, True
        def json(self): return self.payload
        def raise_for_status(self): pass

    def fake_get(url, **kw):
        if url.endswith("/internal/logs"):
            return R(text="step %d" % polls["n"])       # server keeps moving
        polls["n"] += 1
        if polls["n"] < 30:                             # ~30 min of polling
            return R({})
        return R({"pid": {"status": {"status_str": "success"},
                          "outputs": {"9": {"images": []}}}})

    monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
    monkeypatch.setattr(comfy_mod.requests, "post", lambda *a, **k: R({"prompt_id": "pid"}))
    monkeypatch.setattr(comfy_mod.ComfyClient, "_download_outputs", lambda self, o, d: ["ok"])

    # timeout_s is 10 minutes; the loop runs far longer than that in wall clock
    out = comfy_mod.ComfyClient("http://x", timeout_s=600).run_workflow(wf, {}, tmp_path)
    assert out == ["ok"] and clock["t"] > 600


def test_comfy_wait_still_gives_up_on_a_stalled_prompt(monkeypatch, tmp_path):
    import pipeline.adapters.comfy as comfy_mod

    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "X", "inputs": {}}}))
    clock = {"t": 0.0}
    monkeypatch.setattr(comfy_mod.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 60))
    monkeypatch.setattr(comfy_mod.time, "time", lambda: clock["t"])

    class R:
        def __init__(self, payload=None, text=""):
            self.payload, self.text, self.ok = payload, text, True
        def json(self): return self.payload
        def raise_for_status(self): pass

    monkeypatch.setattr(comfy_mod.requests, "get",
                        lambda url, **kw: R({}, text="frozen"))   # never advances
    monkeypatch.setattr(comfy_mod.requests, "post", lambda *a, **k: R({"prompt_id": "pid"}))
    with pytest.raises(TimeoutError, match="no progress"):
        comfy_mod.ComfyClient("http://x", timeout_s=600).run_workflow(wf, {}, tmp_path)
