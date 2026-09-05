extends Node
class_name CloakPhysics

# Verlet cloth on the cloak bone chain the rig builds (Cloak.0 .. Cloak.N).
#
# The alternative was baking a Blender cloth sim into the mesh, which produces
# one fixed swirl that plays back identically whatever the character does. A
# solver run at runtime does the thing that actually reads as cloth: the cloak
# lags when you accelerate, keeps going when you stop, whips when you turn, and
# settles differently every time because it starts from wherever it happened to
# be.
#
# Positions are solved in WORLD space and written back as bone rotations, so the
# cloth is unaffected by how the character is oriented -- solving in the
# skeleton's local space makes gravity rotate with the character, and the cloak
# swings outward when you turn instead of hanging down.

@export var gravity := 9.0
@export var stiffness := 0.55       # 0 loose rag, 1 rigid stick
@export var damping := 0.86         # velocity kept per step; lower is heavier cloth
@export var inertia := 1.15         # how hard the body's motion drags the cloth
@export var max_swing_deg := 62.0   # keeps the cloak off the character's face
@export var wind := Vector3.ZERO
@export var solver_iterations := 4  # constraint passes per step

var _skel: Skeleton3D = null
var _bones: PackedInt32Array = []
var _rest_len: PackedFloat32Array = []
var _pos: Array[Vector3] = []       # world-space particle positions
var _prev: Array[Vector3] = []
var _body_prev := Vector3.ZERO
var _ready_to_solve := false


func setup(skeleton: Skeleton3D) -> void:
	_skel = skeleton
	if _skel == null:
		return
	# collect the chain in order; the rig names them Cloak.0 downward
	var i := 0
	while true:
		var idx := _skel.find_bone("Cloak.%d" % i)
		if idx < 0:
			break
		_bones.append(idx)
		i += 1
	if _bones.size() < 2:
		return
	for b in _bones:
		var world := _skel.global_transform * _skel.get_bone_global_pose(b)
		_pos.append(world.origin)
		_prev.append(world.origin)
	for j in range(_bones.size() - 1):
		_rest_len.append(_pos[j].distance_to(_pos[j + 1]))
	_body_prev = _skel.global_transform.origin
	_ready_to_solve = true


func _physics_process(delta: float) -> void:
	if not _ready_to_solve or delta <= 0.0:
		return
	var root_world := _skel.global_transform * _skel.get_bone_global_pose(_bones[0])
	var anchor := root_world.origin
	var body_vel := (_skel.global_transform.origin - _body_prev) / delta
	_body_prev = _skel.global_transform.origin

	# --- integrate ---------------------------------------------------------
	# The body's own motion is injected as a drag on every particle. Without it
	# the cloak only responds to gravity and hangs dead still while the
	# character sprints, which is the single most obvious tell of fake cloth.
	var accel := Vector3(0.0, -gravity, 0.0) + wind - body_vel * inertia
	for j in range(_pos.size()):
		if j == 0:
			continue                      # the top link is pinned to the shoulders
		var velocity := (_pos[j] - _prev[j]) * damping
		_prev[j] = _pos[j]
		_pos[j] += velocity + accel * delta * delta

	# --- constrain ---------------------------------------------------------
	for _pass in range(solver_iterations):
		_pos[0] = anchor
		for j in range(_pos.size() - 1):
			var a := _pos[j]
			var b := _pos[j + 1]
			var d := b - a
			var length := d.length()
			if length < 0.0001:
				continue
			var correction := d * (1.0 - _rest_len[j] / length) * stiffness
			# the upper particle of a pair is the more constrained one: pulling
			# both equally lets the whole chain drift off the shoulders
			if j > 0:
				_pos[j] += correction * 0.5
				_pos[j + 1] -= correction * 0.5
			else:
				_pos[j + 1] -= correction
		_limit_swing(anchor)

	_write_back()


func _limit_swing(anchor: Vector3) -> void:
	# Cloth solved without limits will happily pass through the wearer. A cone
	# around straight-down is cheap and enough: the cloak hangs behind and to
	# the sides, and cannot reach round to the chest.
	var down := Vector3.DOWN
	var limit := deg_to_rad(max_swing_deg)
	for j in range(1, _pos.size()):
		var from := _pos[j - 1] if j > 1 else anchor
		var d := _pos[j] - from
		var length := d.length()
		if length < 0.0001:
			continue
		var dir := d / length
		var angle := dir.angle_to(down)
		if angle > limit:
			var axis := down.cross(dir)
			if axis.length() < 0.0001:
				continue
			dir = down.rotated(axis.normalized(), limit)
			_pos[j] = from + dir * length


func _write_back() -> void:
	# Turn solved world positions back into bone rotations: for each link, the
	# rotation that takes its rest direction onto the direction the solver found.
	#
	# The solved direction is in skeleton space and a pose rotation is relative
	# to the bone's PARENT, so it has to be converted before it is written. It
	# used to be written raw, which meant every link in the chain re-applied its
	# parent's rotation on top of its own; six links compounding turned the
	# cloak into a cone that stood out around the character like a ballgown.
	# Parents are solved before children here, so the parent's global pose is
	# already the one this frame will use.
	var inv := _skel.global_transform.affine_inverse()
	for j in range(_bones.size() - 1):
		var bone := _bones[j]
		var rest := _skel.get_bone_rest(bone)
		var local_a := inv * _pos[j]
		var local_b := inv * _pos[j + 1]
		var want := (local_b - local_a)
		if want.length() < 0.0001:
			continue
		var parent := _skel.get_bone_parent(bone)
		if parent >= 0:
			want = _skel.get_bone_global_pose(parent).basis.inverse() * want
		want = want.normalized()
		var have := (rest.basis * Vector3.UP).normalized()
		var axis := have.cross(want)
		if axis.length() < 0.0001:
			_skel.set_bone_pose_rotation(bone, rest.basis.get_rotation_quaternion())
			continue
		var q := Quaternion(axis.normalized(), have.angle_to(want))
		_skel.set_bone_pose_rotation(bone, q * rest.basis.get_rotation_quaternion())


## Nudge the cloth, for a hit reaction or a landing.
func impulse(world_dir: Vector3, strength: float) -> void:
	if not _ready_to_solve:
		return
	for j in range(1, _pos.size()):
		var falloff := float(j) / float(_pos.size() - 1)   # the hem moves most
		_pos[j] += world_dir * strength * falloff
