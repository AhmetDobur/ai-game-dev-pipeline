# Walk + look for ONE local player. Dropped in by scaffold.py only when the coder
# LLM did not write a usable player.gd, so a run always ships something playable
# instead of a world nobody can move in.
extends CharacterBody3D

## Gamepad this player reads. Two pads on one PC drive two players without any
## per-player input actions -- Godot addresses joypads by device index directly.
## Every seat also has a keyboard fallback, so split-screen is playable on one
## keyboard when a pad is missing: seat 0 is WASD+mouse, seat 1 is arrows+IJKL.
@export var device: int = 0
## Camera to drive. In split-screen the camera lives inside this player's
## SubViewport and therefore CANNOT be a child of the player, so its transform is
## carried across each frame instead.
@export var camera_path: NodePath

const SPEED := 4.5
const RUN_SPEED := 8.0
const LOOK_SPEED := 2.5
const KEY_LOOK_SPEED := 2.0
const STICK_DEADZONE := 0.2
const RUN_STICK := 0.75        # push the stick past this and the character runs
const GRAVITY := 18.0
const PITCH_LIMIT := 1.2
const MOUSE_SENS := 0.0022

@onready var _pivot: Node3D = $CamPivot
@onready var _spring: SpringArm3D = $CamPivot/SpringArm3D
@onready var _mount: Node3D = $CamPivot/SpringArm3D/CamMount
@onready var _camera: Camera3D = get_node_or_null(camera_path) as Camera3D

var _anim: AnimationPlayer = null
var _combat: CombatController = null
var _cloak: CloakPhysics = null
var _pad_left := false      # edge detection for pad buttons, which have no
var _pad_right := false     # just_pressed equivalent per device
var _clip := ""
var _keyboard := true


func _ready() -> void:
	# without this the boom would sweep through the level: looking up past ~35
	# degrees buries a fixed 3.5 m camera under the floor slab, and turning near
	# a wall pushes it through the masonry
	_spring.add_excluded_object(get_rid())
	_refresh_seat()
	Input.joy_connection_changed.connect(func(_d, _c): _refresh_seat())
	if device == 0:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	# the rigged character.glb carries idle/walk/run as named animations; without
	# this the character slides through the hall frozen in its rest pose
	_anim = find_child("AnimationPlayer", true, false)
	if _anim:
		for a in _anim.get_animation_list():
			_anim.get_animation(a).loop_mode = Animation.LOOP_LINEAR
	# after the loop flags above, so it can clear them on the strike clips: a
	# punch that loops is a windmill
	_combat = CombatController.new()
	add_child(_combat)
	_combat.setup(_anim)
	var skel := find_child("Skeleton3D", true, false) as Skeleton3D
	if skel:
		_cloak = CloakPhysics.new()
		# a SkeletonModifier3D only runs when it is a child of the skeleton
		skel.add_child(_cloak)
		_cloak.setup(skel)
		# a landed punch throws the shoulders, and cloth that ignores the hit
		# is the moment the illusion drops
		_combat.strike_landed.connect(func(_m, _d):
			_cloak.impulse(-global_transform.basis.z, 0.06))


func _refresh_seat() -> void:
	# seat 0 is always keyboard-capable; seat 1 falls back to keys only while no
	# pad is plugged in for it
	_keyboard = device == 0 or not (device in Input.get_connected_joypads())


func _input(event: InputEvent) -> void:
	# _input, not _unhandled_input: the split-screen Controls sit above the
	# players in the input chain and a captured-mouse look must not depend on
	# whether a container let the event through.
	if device != 0:
		return
	if event is InputEventMouseMotion \
			and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * MOUSE_SENS)
		_pitch(-event.relative.y * MOUSE_SENS)
	elif event.is_action_pressed("ui_cancel"):
		# Esc releases the cursor -- otherwise a windowed build traps the mouse
		# with no way to reach the close button.
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	elif event is InputEventMouseButton and event.pressed \
			and Input.get_mouse_mode() == Input.MOUSE_MODE_VISIBLE:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


func _pitch(delta_x: float) -> void:
	_pivot.rotation.x = clampf(_pivot.rotation.x + delta_x, -PITCH_LIMIT, PITCH_LIMIT)


func _play(clip: String) -> void:
	if _anim and clip != _clip and _anim.has_animation(clip):
		_anim.play(clip, 0.25)   # cross-fade so walk<->run does not snap
		_clip = clip


func _stick(axis_x: int, axis_y: int) -> Vector2:
	var v := Vector2(Input.get_joy_axis(device, axis_x),
			Input.get_joy_axis(device, axis_y))
	# radial deadzone: per-axis thresholds let a centred stick creep diagonally
	return Vector2.ZERO if v.length() < STICK_DEADZONE else v


func _keys(prefix: String, a: String, b: String, c: String, d: String) -> Vector2:
	return Input.get_vector(prefix + a, prefix + b, prefix + c, prefix + d)


func _physics_process(delta: float) -> void:
	var prefix := "" if device == 0 else "p2_"

	# Q is the left hand, E the right; on a pad, X and B. Read every frame so a
	# press during a strike is buffered rather than dropped.
	if Input.is_action_just_pressed(prefix + "attack_left") \
			or (not _keyboard and Input.is_joy_button_pressed(device, JOY_BUTTON_X)
				and not _pad_left):
		_combat.press("q")
	if Input.is_action_just_pressed(prefix + "attack_right") \
			or (not _keyboard and Input.is_joy_button_pressed(device, JOY_BUTTON_B)
				and not _pad_right):
		_combat.press("e")
	_pad_left = not _keyboard and Input.is_joy_button_pressed(device, JOY_BUTTON_X)
	_pad_right = not _keyboard and Input.is_joy_button_pressed(device, JOY_BUTTON_B)
	_combat.tick(delta)

	var look := _stick(JOY_AXIS_RIGHT_X, JOY_AXIS_RIGHT_Y)
	if look == Vector2.ZERO and _keyboard and device != 0:
		look = _keys(prefix, "look_left", "look_right", "look_up", "look_down") \
				* (KEY_LOOK_SPEED / LOOK_SPEED)
	rotate_y(-look.x * LOOK_SPEED * delta)
	_pitch(-look.y * LOOK_SPEED * delta)

	# Run is a STICK-THROW test, so it may only be applied to stick input:
	# Input.get_vector always returns magnitude 1.0 for a key press, which made
	# the keyboard seat sprint permanently and put walk speed out of reach.
	var stick := _stick(JOY_AXIS_LEFT_X, JOY_AXIS_LEFT_Y)
	var move := stick
	var running := stick.length() > RUN_STICK
	if move == Vector2.ZERO and _keyboard:
		move = _keys(prefix, "move_left", "move_right", "move_forward", "move_back")
		running = Input.is_key_pressed(KEY_SHIFT if device == 0 else KEY_CTRL)

	var dir := (transform.basis * Vector3(move.x, 0.0, move.y))
	dir.y = 0.0
	dir = dir.normalized() if dir.length() > 0.001 else Vector3.ZERO
	velocity.x = dir.x * (RUN_SPEED if running else SPEED)
	velocity.z = dir.z * (RUN_SPEED if running else SPEED)
	velocity.y = 0.0 if is_on_floor() else velocity.y - GRAVITY * delta
	move_and_slide()

	# A strike owns the body until it recovers. Without this the locomotion clip
	# is re-selected every frame and immediately overwrites the punch, so the
	# character appears to ignore the input entirely.
	if not _combat.locks_movement():
		if dir == Vector3.ZERO:
			_play("idle")
		else:
			_play("run" if running else "walk")
	else:
		_clip = ""      # so the locomotion clip re-plays cleanly on recovery

	# the camera renders inside a SubViewport, so it cannot be a child of the
	# player; carry the spring arm's collision-corrected tip across every frame
	if _camera:
		_camera.global_transform = _mount.global_transform
