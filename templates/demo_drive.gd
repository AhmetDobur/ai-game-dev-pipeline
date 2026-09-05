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
var _fill: OmniLight3D = null
var _rim: OmniLight3D = null
var _body: Node3D = null
var _angle := 0.0


# PIPELINE_DEMO=light  -> lights only; you keep the controls and the game camera
# PIPELINE_DEMO=drive|1 -> lights, demo camera and the scripted timeline, which
#                          is what a --write-movie recording wants
var _drive := false


func _ready() -> void:
	var mode := OS.get_environment("PIPELINE_DEMO")
	if mode == "":
		set_process(false)
		return
	_drive = mode != "light"
	if not _drive:
		set_process(false)
	print("[demo] mode=%s" % mode)
	call_deferred("_rig_camera")


func _rig_camera() -> void:
	_body = _find_body(get_tree().root)
	if _body == null:
		push_warning("[demo] no player body found; leaving the game camera alone")
		return
	var root := get_tree().current_scene
	if _drive:
		_cam = Camera3D.new()
		_cam.fov = 55.0
		root.add_child(_cam)
		_cam.current = true
	# Key/fill/rim rather than one hot spot. A single lamp on a dark robe either
	# leaves it a silhouette or blows the specular out to white; the fill opens
	# the shadow side and the rim separates the cloak from a black hall.
	_key = SpotLight3D.new()
	_key.light_energy = 5.0
	_key.spot_range = 16.0
	_key.spot_angle = 50.0
	_key.light_color = Color(1.0, 0.94, 0.86)
	_key.shadow_enabled = true
	root.add_child(_key)

	_fill = OmniLight3D.new()
	_fill.light_energy = 2.2
	_fill.omni_range = 9.0
	_fill.light_color = Color(0.72, 0.80, 1.0)
	root.add_child(_fill)

	_rim = OmniLight3D.new()
	_rim.light_energy = 3.0
	_rim.omni_range = 9.0
	_rim.light_color = Color(1.0, 0.86, 0.70)
	root.add_child(_rim)

	# lift the hall itself off black so the character is not floating in a void
	var env := _find_env(get_tree().root)
	if env != null and env.environment != null:
		env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
		env.environment.ambient_light_color = Color(0.62, 0.60, 0.66)
		env.environment.ambient_light_energy = 0.55
	_place_camera()


func _find_env(n: Node) -> WorldEnvironment:
	if n is WorldEnvironment:
		return n
	for c in n.get_children():
		var f := _find_env(c)
		if f != null:
			return f
	return null


func _find_body(n: Node) -> Node3D:
	if n is CharacterBody3D:
		return n
	for c in n.get_children():
		var f := _find_body(c)
		if f != null:
			return f
	return null


func _physics_process(_d: float) -> void:
	if not _drive:
		_place_camera()          # lights track the player even with no timeline


func _place_camera() -> void:
	if _body == null or _key == null:
		return
	# behind the shoulder while running (the cloak lives on the back), swinging
	# round to three-quarter front for the striking section
	var want: float = 0.0 if (_t < 7.0 or _t > 15.6) else 2.4
	_angle = lerp(_angle, want, 0.03)
	var base := _body.global_position
	var yaw := _body.global_rotation.y + _angle
	var offset := Vector3(sin(yaw), 0.0, cos(yaw)) * CAM_DIST
	var eye := base + offset + Vector3(0.0, CAM_HEIGHT, 0.0)
	if _cam != null:
		_cam.global_position = eye
		_cam.look_at(base + Vector3(0.0, AIM_HEIGHT, 0.0), Vector3.UP)
	var aim := base + Vector3(0.0, AIM_HEIGHT, 0.0)
	_key.global_position = eye + Vector3(0.0, 1.2, 0.0)
	_key.look_at(aim, Vector3.UP)
	# fill opposite the key, rim behind: both ride the same orbit so the shaping
	# holds as the camera swings round for the striking section
	var side := Vector3(sin(yaw + 2.0), 0.0, cos(yaw + 2.0)) * 2.6
	_fill.global_position = base + side + Vector3(0.0, 1.5, 0.0)
	var behind := Vector3(sin(yaw + PI), 0.0, cos(yaw + PI)) * 2.4
	_rim.global_position = base + behind + Vector3(0.0, 2.1, 0.0)


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
