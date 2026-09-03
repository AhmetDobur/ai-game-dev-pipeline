# Dropped into a generated project by scaffold.py and run once, headless-windowed,
# after the build. The pipeline had no way to see its own game: every run shipped
# an .exe nobody had looked at. This writes ONE screenshot next to the build so a
# run is self-documenting and a black or empty world is visible without launching.
extends Node3D

const SHOT_FRAME := 20

var _f := 0
var _cam: Camera3D = null

func _ready() -> void:
	# Inert in the shipped game: without PIPELINE_SHOT set this node does nothing
	# at all, so attaching it to the world scene cannot make a player's build
	# quit after twenty frames.
	if OS.get_environment("PIPELINE_SHOT") == "":
		set_process(false)
		return
	# Do not disturb a camera the game already has; only add one if none exists.
	_cam = _find_camera(get_tree().root)
	if _cam == null:
		_cam = Camera3D.new()
		add_child(_cam)
		_cam.global_position = Vector3(0, 4.5, -18.0)
		_cam.look_at(Vector3(0, 4.0, 12.0), Vector3.UP)
		_cam.fov = 75.0
		_cam.current = true

func _find_camera(n: Node) -> Camera3D:
	if n is Camera3D:
		return n
	for c in n.get_children():
		var f := _find_camera(c)
		if f != null:
			return f
	return null

func _process(_delta: float) -> void:
	_f += 1
	if _f < SHOT_FRAME:
		return
	var path := OS.get_environment("PIPELINE_SHOT")
	if path == "":
		path = "user://world_shot.png"
	var img := get_viewport().get_texture().get_image()
	img.save_png(path)
	print("PIPELINE_SHOT_SAVED ", path)
	get_tree().quit()
