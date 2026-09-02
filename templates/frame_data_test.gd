# Frame-data grader — shipped by the pipeline, NOT written by the coder model.
# Runs headless:  godot --headless --path <game> --script res://tests/frame_data_test.gd
#
# Contract: the coder model must implement res://scripts/combat_sim.gd exposing:
#   setup(move: String) -> void   # two fighters at the move's specced range, 60fps fixed step
#   press(move: String) -> void   # buffer the input for the next frame
#   step() -> void                # advance exactly one frame
#   hitbox_active() -> bool       # attacker's hitbox live this frame
#   opponent_in_hitstun() -> bool
#   opponent_offset() -> Vector2  # opponent displacement since setup()
#
# Frame data lives in res://frame_data.json:
#   {"punch": {"startup": 8, "active": 3, "hitstun": 18,
#              "knockback": [40, 0], "tolerance": 2}, ...}
extends SceneTree

func _init() -> void:
	var file := FileAccess.open("res://frame_data.json", FileAccess.READ)
	if file == null:
		push_error("frame_data.json missing")
		quit(1)
		return
	var table: Dictionary = JSON.parse_string(file.get_as_text())
	var sim_script := load("res://scripts/combat_sim.gd")
	if sim_script == null:
		push_error("scripts/combat_sim.gd missing or does not parse")
		quit(1)
		return
	var failures := 0
	for move in table.keys():
		failures += _grade_move(sim_script, move, table[move])
	if failures == 0:
		print("FRAME_DATA_OK")
		quit(0)
	else:
		print("FRAME_DATA_FAILURES=%d" % failures)
		quit(1)

func _grade_move(sim_script: GDScript, move: String, spec: Dictionary) -> int:
	var sim = sim_script.new()
	sim.setup(move)
	sim.press(move)
	var failures := 0
	var tol: float = float(spec.get("tolerance", 0))

	# startup: frames until the hitbox goes active
	var frame := 0
	while not sim.hitbox_active() and frame < 600:
		sim.step()
		frame += 1
	if absf(frame - float(spec["startup"])) > tol:
		print("FAIL %s startup: expected %s got %d" % [move, spec["startup"], frame])
		failures += 1

	# active: frames the hitbox stays live
	var active := 0
	while sim.hitbox_active() and active < 600:
		sim.step()
		active += 1
	if absf(active - float(spec["active"])) > tol:
		print("FAIL %s active: expected %s got %d" % [move, spec["active"], active])
		failures += 1

	# hitstun: frames the opponent stays stunned after contact
	var stun := 0
	while sim.opponent_in_hitstun() and stun < 600:
		sim.step()
		stun += 1
	if absf(stun - float(spec["hitstun"])) > tol:
		print("FAIL %s hitstun: expected %s got %d" % [move, spec["hitstun"], stun])
		failures += 1

	# knockback: opponent displacement once everything settles
	if spec.has("knockback"):
		var kb: Array = spec["knockback"]
		var off: Vector2 = sim.opponent_offset()
		if absf(off.x - float(kb[0])) > maxf(tol, 1.0) or absf(off.y - float(kb[1])) > maxf(tol, 1.0):
			print("FAIL %s knockback: expected %s got (%s, %s)" % [move, str(kb), off.x, off.y])
			failures += 1
	return failures
