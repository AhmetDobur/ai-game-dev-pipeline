# Settings grader -- shipped by the pipeline, NOT written by the coder model.
# Runs headless:  godot --headless --path <game> --script res://tests/settings_test.gd
#
# Covers the two things in settings.gd that can silently do the wrong thing: the
# ConfigFile round-trip (a slider that forgets on restart looks identical to one
# that works, until you restart) and the rebind conflict rule (a key bound twice
# fires two actions and is only noticed mid-fight).
extends SceneTree

const KEY_W := 87
const KEY_J := 74      # ships as p2_look_left, so binding it to P1 must free it

var _fails := 0


func _check(ok: bool, what: String) -> void:
	if not ok:
		_fails += 1
		push_error("FAIL: " + what)
	else:
		print("ok: " + what)


func _init() -> void:
	# Start from the shipped defaults. settings.cfg lives in user:// and outlives
	# the run, so without this the test grades whatever the last run happened to
	# save and fails on the bindings it just checked in.
	DirAccess.remove_absolute(ProjectSettings.globalize_path("user://settings.cfg"))

	var settings = load("res://scripts/settings.gd").new()
	get_root().add_child(settings)
	# _ready fires when the tree next processes, not on add_child, and _init runs
	# before the first frame. Without this wait the node under test is still
	# uninitialised -- _defaults is empty, and reset_bindings quietly does
	# nothing, which reads exactly like a broken reset.
	await process_frame

	# --- rebind takes the key away from whoever held it --------------------
	_check(settings.keycode_of("move_forward") == KEY_W, "P1 forward ships as W")
	_check(settings.keycode_of("p2_look_left") == KEY_J, "P2 look-left ships as J")
	settings.rebind("move_forward", KEY_J)
	_check(settings.keycode_of("move_forward") == KEY_J, "forward took J")
	_check(settings.keycode_of("p2_look_left") == 0, "J was taken off P2 look-left")

	# --- values survive a write and read ------------------------------------
	settings.set_volume("SFX", 0.42)
	settings.set_brightness(-0.5)
	settings.save_settings()
	settings.set_volume("SFX", 1.0)
	settings.set_brightness(0.0)
	settings.load_settings()
	_check(abs(settings.volume["SFX"] - 0.42) < 0.001, "SFX volume round-tripped")
	_check(abs(settings.brightness - (-0.5)) < 0.001, "brightness round-tripped")
	_check(settings.keycode_of("move_forward") == KEY_J, "binding round-tripped")

	# --- reset puts the shipped keys back -----------------------------------
	settings.reset_bindings()
	_check(settings.keycode_of("move_forward") == KEY_W, "reset restored W")
	_check(settings.keycode_of("p2_look_left") == KEY_J, "reset restored P2 look-left")

	# a bad volume must not become a bad bus gain
	settings.set_volume("Master", 5.0)
	_check(settings.volume["Master"] == 1.0, "volume clamped at 1.0")
	settings.set_brightness(-9.0)
	_check(settings.brightness == -1.0, "brightness clamped at -1.0")

	print("settings_test: %d failure(s)" % _fails)
	quit(1 if _fails else 0)
