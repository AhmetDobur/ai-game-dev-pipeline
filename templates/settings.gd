extends Node

# Player settings: audio levels, screen brightness, key bindings.
#
# An autoload rather than something the menu owns, because two of the three have
# to hold while the menu is not on screen. Volume and brightness set in the menu
# and then forgotten the moment the fight starts would be a settings screen that
# does not settle anything.
#
# Everything lands in user://settings.cfg through ConfigFile -- Godot's own
# ini reader/writer, so there is no format to invent and nothing to parse.

const PATH := "user://settings.cfg"

# Audio is split three ways because the sources are genuinely different and
# people balance them differently: the strike sounds are constant and loud, the
# voice lines are occasional and want to sit above them.
const BUSES := ["Master", "SFX", "Voice"]

# Seat 1 and seat 2, in the order they are shown. Labels are here rather than in
# the menu so that adding an action to the input map is a one-line change in one
# file.
const BINDABLE := [
	["Player 1", {
		"move_forward": "Forward", "move_back": "Back",
		"move_left": "Left", "move_right": "Right",
		"attack_left": "Left hand", "attack_right": "Right hand",
	}],
	["Player 2", {
		"p2_move_forward": "Forward", "p2_move_back": "Back",
		"p2_move_left": "Left", "p2_move_right": "Right",
		"p2_look_up": "Look up", "p2_look_down": "Look down",
		"p2_look_left": "Look left", "p2_look_right": "Look right",
		"p2_attack_left": "Left hand", "p2_attack_right": "Right hand",
	}],
]

var volume := {"Master": 0.8, "SFX": 0.8, "Voice": 1.0}
var brightness := 0.0          # -1 pitch black .. 0 untouched .. +1 washed out

var _defaults := {}            # action -> physical keycode as the project shipped it
var _veil: ColorRect = null


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	for group in BINDABLE:
		for action in group[1]:
			_defaults[action] = keycode_of(action)
	_make_buses()
	_make_veil()
	load_settings()


# --- audio -------------------------------------------------------------------

func _make_buses() -> void:
	# Created in code instead of shipping a .tres bus layout: three buses wired
	# to Master is less configuration than a binary file somebody has to open
	# Godot to inspect.
	for name in BUSES:
		if name == "Master" or AudioServer.get_bus_index(name) != -1:
			continue
		var i := AudioServer.bus_count
		AudioServer.add_bus(i)
		AudioServer.set_bus_name(i, name)
		AudioServer.set_bus_send(i, "Master")


func set_volume(bus: String, value: float) -> void:
	volume[bus] = clampf(value, 0.0, 1.0)
	var i := AudioServer.get_bus_index(bus)
	if i == -1:
		return
	# Silence has no decibel value, so the bottom of the slider mutes rather
	# than trying to express -inf dB.
	AudioServer.set_bus_mute(i, volume[bus] <= 0.001)
	AudioServer.set_bus_volume_db(i, linear_to_db(maxf(volume[bus], 0.001)))


# --- brightness --------------------------------------------------------------

func _make_veil() -> void:
	# A full-screen rect on a layer above everything, rather than an Environment
	# adjustment: the menu has no WorldEnvironment to adjust, and a brightness
	# setting that only works once a fight has started is a trap.
	var layer := CanvasLayer.new()
	layer.layer = 128
	add_child(layer)
	_veil = ColorRect.new()
	_veil.set_anchors_preset(Control.PRESET_FULL_RECT)
	_veil.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var mat := CanvasItemMaterial.new()
	_veil.material = mat
	layer.add_child(_veil)


func set_brightness(value: float) -> void:
	brightness = clampf(value, -1.0, 1.0)
	if _veil == null:
		return
	# One rect covers both directions: black over the top darkens, white added
	# on top lifts. Capped well under 1.0 so the extremes stay playable instead
	# of handing the player a way to render the game unusable.
	var mat := _veil.material as CanvasItemMaterial
	if brightness >= 0.0:
		mat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
		_veil.color = Color(1, 1, 1, brightness * 0.35)
	else:
		mat.blend_mode = CanvasItemMaterial.BLEND_MODE_MIX
		_veil.color = Color(0, 0, 0, -brightness * 0.75)


# --- key bindings ------------------------------------------------------------

func keycode_of(action: String) -> int:
	if not InputMap.has_action(action):
		return 0
	for ev in InputMap.action_get_events(action):
		if ev is InputEventKey:
			return ev.physical_keycode
	return 0


func label_for(action: String) -> String:
	var code := keycode_of(action)
	if code == 0:
		return "--"
	# Physical keycodes are stored so a binding survives a layout change, but a
	# player on AZERTY has to be shown the letter actually printed on the key.
	# The headless server cannot answer that and logs an error per call, so it
	# is asked only where it can be: a build with no display shows the US name.
	if DisplayServer.get_name() != "headless":
		code = DisplayServer.keyboard_get_keycode_from_physical(code)
	return OS.get_keycode_string(code)


func rebind(action: String, keycode: int) -> void:
	if not InputMap.has_action(action):
		return
	# A key can only mean one thing. Whatever else held it gives it up, or the
	# rebind silently creates a key that fires two actions at once.
	for group in BINDABLE:
		for other in group[1]:
			if other != action and keycode_of(other) == keycode:
				_assign(other, 0)
	_assign(action, keycode)


func _assign(action: String, keycode: int) -> void:
	InputMap.action_erase_events(action)
	if keycode == 0:
		return
	var ev := InputEventKey.new()
	ev.physical_keycode = keycode
	InputMap.action_add_event(action, ev)


func reset_bindings() -> void:
	for action in _defaults:
		_assign(action, _defaults[action])


# --- persistence -------------------------------------------------------------

func load_settings() -> void:
	var cfg := ConfigFile.new()
	# A missing file on first run is the normal case, not a failure: applying
	# the defaults gives the buses and the veil their initial state either way.
	var ok := cfg.load(PATH) == OK
	for bus in BUSES:
		set_volume(bus, cfg.get_value("audio", bus, volume[bus]) if ok else volume[bus])
	set_brightness(cfg.get_value("video", "brightness", 0.0) if ok else 0.0)
	if not ok:
		return
	for group in BINDABLE:
		for action in group[1]:
			var code: int = cfg.get_value("keys", action, _defaults.get(action, 0))
			_assign(action, code)


func save_settings() -> void:
	var cfg := ConfigFile.new()
	for bus in BUSES:
		cfg.set_value("audio", bus, volume[bus])
	cfg.set_value("video", "brightness", brightness)
	for group in BINDABLE:
		for action in group[1]:
			cfg.set_value("keys", action, keycode_of(action))
	cfg.save(PATH)
