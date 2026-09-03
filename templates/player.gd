# Walk + look for ONE local player. Dropped in by scaffold.py only when the coder
# LLM did not write a usable player.gd, so a run always ships something playable
# instead of a world nobody can move in.
extends CharacterBody3D

## Gamepad this player reads. Two pads on one PC drive two players without any
## per-player input actions -- Godot addresses joypads by device index directly.
## Device 0 also accepts the keyboard, so the game is playable with no pad at all.
@export var device: int = 0
## Camera to drive. In split-screen the camera lives inside this player's
## SubViewport and therefore CANNOT be a child of the player, so its transform is
## carried across each frame instead.
@export var camera_path: NodePath

const SPEED := 4.5
const LOOK_SPEED := 2.5
const STICK_DEADZONE := 0.2
const GRAVITY := 18.0
const PITCH_LIMIT := 1.2
const CAM_OFFSET := Vector3(0, 0.4, 3.5)
const MOUSE_SENS := 0.0022

@onready var _pivot: Node3D = $CamPivot
@onready var _camera: Camera3D = get_node_or_null(camera_path) as Camera3D


func _ready() -> void:
	# Player 1 is the keyboard+mouse seat. Without this the game is walk-only:
	# WASD moves but nothing turns, so you can never look at the room you are in.
	if device == 0:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


func _input(event: InputEvent) -> void:
	# _input, not _unhandled_input: the split-screen Controls sit above the
	# players in the input chain and a captured-mouse look must not depend on
	# whether a container let the event through.
	if device != 0:
		return
	if event is InputEventMouseMotion \
			and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * MOUSE_SENS)
		_pivot.rotation.x = clampf(_pivot.rotation.x - event.relative.y * MOUSE_SENS,
				-PITCH_LIMIT, PITCH_LIMIT)
	elif event.is_action_pressed("ui_cancel"):
		# Esc releases the cursor -- otherwise a windowed build traps the mouse
		# with no way to reach the close button.
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	elif event is InputEventMouseButton and event.pressed \
			and Input.get_mouse_mode() == Input.MOUSE_MODE_VISIBLE:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


func _stick(axis_x: int, axis_y: int) -> Vector2:
	var v := Vector2(Input.get_joy_axis(device, axis_x),
			Input.get_joy_axis(device, axis_y))
	# radial deadzone: per-axis thresholds let a centred stick creep diagonally
	return Vector2.ZERO if v.length() < STICK_DEADZONE else v


func _physics_process(delta: float) -> void:
	var look := _stick(JOY_AXIS_RIGHT_X, JOY_AXIS_RIGHT_Y)
	rotate_y(-look.x * LOOK_SPEED * delta)
	_pivot.rotation.x = clampf(_pivot.rotation.x - look.y * LOOK_SPEED * delta,
			-PITCH_LIMIT, PITCH_LIMIT)

	var move := _stick(JOY_AXIS_LEFT_X, JOY_AXIS_LEFT_Y)
	if device == 0 and move == Vector2.ZERO:
		move = Input.get_vector("move_left", "move_right",
				"move_forward", "move_back")
	var dir := (transform.basis * Vector3(move.x, 0.0, move.y))
	dir.y = 0.0
	dir = dir.normalized() if dir.length() > 0.001 else Vector3.ZERO
	velocity.x = dir.x * SPEED
	velocity.z = dir.z * SPEED
	velocity.y = 0.0 if is_on_floor() else velocity.y - GRAVITY * delta
	move_and_slide()

	if _camera:
		_camera.global_transform = _pivot.global_transform.translated_local(CAM_OFFSET)
