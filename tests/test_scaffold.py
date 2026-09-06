from pathlib import Path

from pipeline.scaffold import scaffold


def test_scaffold_writes_runnable_skeleton(tmp_path):
    game = tmp_path / "game"
    (game / "scripts").mkdir(parents=True)
    (game / "scripts" / "player.gd").write_text("extends CharacterBody3D\n")
    char = tmp_path / "idle.glb"; char.write_bytes(b"g" * 10)
    env = tmp_path / "hall.glb"; env.write_bytes(b"g" * 10)
    mesh_src = tmp_path / "raw.glb"; mesh_src.write_bytes(b"g" * 10)

    scaffold(game, 'My "Game"',
             dep_outputs={"r1-anim": [str(char)], "r1-hall": [str(env)],
                          "r1-mesh": [str(mesh_src)], "r1-code": []},
             dep_types={"r1-anim": "rig_animate", "r1-hall": "design_3d",
                        "r1-mesh": "design_3d", "r1-code": "code"},
             dep_specs={"r1-anim": {"mesh_from": "r1-mesh"}})

    proj = (game / "project.godot").read_text()
    # the project opens on the title screen; the world is what FIGHT loads
    assert 'run/main_scene="res://scenes/menu.tscn"' in proj
    assert 'Settings="*res://scripts/settings.gd"' in proj
    menu = (game / "scenes" / "menu.tscn").read_text()
    assert 'world_scene = "res://scenes/world.tscn"' in menu
    assert (game / "scripts" / "menu.gd").exists()
    assert (game / "scripts" / "settings.gd").exists()
    assert "move_forward" in proj and '"' + 'My Game' + '"' in proj
    assert (game / "export_presets.cfg").exists()
    world = (game / "scenes" / "world.tscn").read_text()
    # character = the rig output, env = the unconsumed design_3d, raw mesh excluded
    assert "assets/anim/idle.glb" in world
    assert "assets/hall/hall.glb" in world
    assert "raw.glb" not in world
    assert 'path="res://scripts/player.gd"' in world
    # every referenced asset was really copied into the project
    for ref in ("assets/anim/idle.glb", "assets/hall/hall.glb"):
        assert (game / ref).exists()


def test_scaffold_survives_no_assets(tmp_path):
    game = tmp_path / "game"
    scaffold(game, "Empty", {}, {}, {})
    assert (game / "project.godot").exists()
    assert (game / "scenes" / "world.tscn").exists()


def test_each_fighter_gets_its_own_voice_folder(tmp_path):
    """Two seats, two voices -- matched on the mesh name, not on finish order.

    The art stage finishes the two characters in whatever order they happen to
    render, so a seat that took its voice from the list position would swap the
    two fighters' grunts between runs on the same project.
    """
    game = tmp_path / "game"
    (game / "scripts").mkdir(parents=True)
    (game / "scripts" / "player.gd").write_text("extends CharacterBody3D\n")
    shadow = tmp_path / "veiled_shadow.glb"; shadow.write_bytes(b"g" * 10)
    pious = tmp_path / "pious_force.glb"; pious.write_bytes(b"g" * 10)

    scaffold(game, "Two",
             dep_outputs={"r1-a": [str(shadow)], "r1-b": [str(pious)]},
             dep_types={"r1-a": "rig_animate", "r1-b": "rig_animate"},
             dep_specs={})

    world = (game / "scenes" / "world.tscn").read_text()
    p1 = world.split('[node name="Player1"')[1].split("[node")[0]
    p2 = world.split('[node name="Player2"')[1].split("[node")[0]
    assert 'voice_dir = "res://audio/voice/veiled_shadow"' in p1
    assert 'voice_dir = "res://audio/voice/pious_force"' in p2
