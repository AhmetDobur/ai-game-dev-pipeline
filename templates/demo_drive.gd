# Drives the player through the whole moveset on a timeline so a recording shows
# what the character can actually do. Without it a capture is a man standing
# still: nobody is pressing keys in a headless render.
#
# Inert unless PIPELINE_DEMO is set, exactly like godot_shot.gd, so shipping it
# in a build cannot take the controls away from a player.
extends Node

# [seconds, action, pressed]. Attacks are tapped, movement is held.
const SCRIPT_ := [
	[0.0,  "", false],                      # 0-2.5s: breathing idle
	[2.5,  "move_forward", true],           # 2.5-6.5s: run, cloak trails
	[6.5,  "move_forward", false],
	[7.5,  "attack_left", true],            # q         -> jab
	[8.5,  "attack_right", true],           # e         -> cross
	[9.6,  "attack_left", true],            # q,e       -> jab, overhand
	[9.95, "attack_right", true],
	[11.2, "attack_right", true],           # e,q       -> cross, left uppercut
	[11.55, "attack_left", true],
	[12.8, "attack_left", true],            # q,q       -> jab, left body shot
	[13.15, "attack_left", true],
	[14.4, "attack_right", true],           # e,e       -> cross, right uppercut
	[14.75, "attack_right", true],
	[16.0, "move_forward", true],           # run out under the swinging cloak
	[19.0, "move_forward", false],
	[20.5, "__quit", false],
]

const TAP_SECONDS := 0.05

# The hall is lit for atmosphere, which makes a dark-robed character a smudge at
# the far end of a colonnade. The demo therefore brings its own camera and key
# light instead of restyling the game: both exist only while PIPELINE_DEMO is
# set, so what ships is exactly what a player gets.
const CAM_DIST := 3.6
const CAM_HEIGHT := 1.5
const AIM_HEIGHT := 0.85

var _t := 0.0
var _i := 0
var _release: Array = []
var _cam: Camera3D = null
var _key: SpotLight3D = null
var _body: Node3D = null
var _angle := 0.0


func _ready() -> void:
	if OS.get_environment("PIPELINE_DEMO") == "":
		set_process(false)
		return
	print("[demo] driving the moveset")
	call_deferred("_rig_camera")


func _rig_camera() -> void:
	_body = _find_body(get_tree().root)
	if _body == null:
		push_warning("[demo] no player body found; leaving the game camera alone")
		return
	var root := get_tree().current_scene
	_cam = Camera3D.new()
	_cam.fov = 55.0
	root.add_child(_cam)
	_cam.current = true
	# a spot riding with the camera: the cloak needs a moving highlight to read
	# as cloth, and a flat ambient lift washes the hall out instead
	_key = SpotLight3D.new()
	_key.light_energy = 3.6
	_key.spot_range = 14.0
	_key.spot_angle = 42.0
	_key.light_color = Color(1.0, 0.94, 0.86)
	root.add_child(_key)
	_place_camera()


func _find_body(n: Node) -> Node3D:
	if n is CharacterBody3D:
		return n
	for c in n.get_children():
		var f := _find_body(c)
		if f != null:
			return f
	return null


func _place_camera() -> void:
	if _cam == null or _body == null:
		return
	# behind the shoulder while running (the cloak lives on the back), swinging
	# round to three-quarter front for the striking section
	var want: float = 0.0 if (_t < 7.0 or _t > 15.6) else 2.4
	_angle = lerp(_angle, want, 0.03)
	var base := _body.global_position
	var yaw := _body.global_rotation.y + _angle
	var offset := Vector3(sin(yaw), 0.0, cos(yaw)) * CAM_DIST
	_cam.global_position = base + offset + Vector3(0.0, CAM_HEIGHT, 0.0)
	_cam.look_at(base + Vector3(0.0, AIM_HEIGHT, 0.0), Vector3.UP)
	_key.global_position = _cam.global_position + Vector3(0.0, 1.2, 0.0)
	_key.look_at(base + Vector3(0.0, AIM_HEIGHT, 0.0), Vector3.UP)


func _process(delta: float) -> void:
	_t += delta
	_place_camera()
	while _i < SCRIPT_.size() and _t >= float(SCRIPT_[_i][0]):
		var action: String = SCRIPT_[_i][1]
		var down: bool = SCRIPT_[_i][2]
		_i += 1
		if action == "__quit":
			get_tree().quit()
			return
		if action == "":
			continue
		if down:
			Input.action_press(action)
			# a movement hold is released by its own later entry; an attack is a
			# tap, so schedule the release here rather than doubling the table
			if action.begins_with("attack_"):
				_release.append([_t + TAP_SECONDS, action])
		else:
			Input.action_release(action)
	for entry in _release.duplicate():
		if _t >= float(entry[0]):
			Input.action_release(entry[1])
			_release.erase(entry)
