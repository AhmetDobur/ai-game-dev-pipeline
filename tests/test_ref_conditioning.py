"""Reference-image conditioning: crop, ref_image repair, scaffold environment."""
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
