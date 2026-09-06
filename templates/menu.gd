extends Control

# Title screen and settings, built in code rather than laid out as a .tscn.
#
# The scaffold's rule is that generated scene files are a wildcard and templated
# ones always parse; a menu is mostly repetition (ten near-identical rebind rows)
# so the smallest thing that always parses is a scene holding one node and a
# script that builds the rest. The palette arrives as exported colours, sampled
# by the scaffold from the same reference image the hall is keyed to, so the
# menu is in the art's colours instead of a guess at them.

@export var shadow := Color("#0b0906")     # near-black ground
@export var stone := Color("#8d8578")      # body text
@export var accent := Color("#6a2f24")     # oxblood, for hover
@export var gold := Color("#c9a227")       # tarnished brass, for the title
@export var title_text := "UNTITLED"
@export var background := ""               # concept art behind the title, if any
@export var world_scene := "res://scenes/world.tscn"

const TITLE_SIZE := 78
const ITEM_SIZE := 26
const LABEL_SIZE := 18

var _main: Control = null
var _settings: Control = null
var _capturing := ""                       # action waiting for a key, "" when idle
var _capture_button: Button = null


func _ready() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)
	var bg := ColorRect.new()
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	bg.color = shadow
	bg.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(bg)
	_build_backdrop()
	_main = _build_main()
	_settings = _build_settings()
	add_child(_main)
	add_child(_settings)
	_settings.hide()
	# The menu is the main scene, so a pipeline run that just boots the project
	# now lands here rather than in the world. Photograph the title screen, then
	# hand over to the world so godot_shot.gd still gets its shot and the world
	# scene is still exercised -- otherwise making the menu the entry point would
	# have quietly stopped the run from ever loading the game it built.
	var shot := OS.get_environment("PIPELINE_SHOT")
	if shot != "":
		var dir := shot.get_base_dir()
		await RenderingServer.frame_post_draw
		get_viewport().get_texture().get_image().save_png(
			dir.path_join("menu_shot.png"))
		# and the settings screen, which is otherwise three clicks from anything
		# automated and so never appeared in a run's own record of itself
		_on_settings()
		await RenderingServer.frame_post_draw
		get_viewport().get_texture().get_image().save_png(
			dir.path_join("settings_shot.png"))
		get_tree().change_scene_to_file(world_scene)


func _build_backdrop() -> void:
	# The run's own environment reference, dimmed, behind the title. A menu with
	# nothing behind it is in the right colours but belongs to no particular
	# game; the concept art the rest of the project was generated from is the one
	# image guaranteed to be on theme, and it costs one file copy.
	if background == "" or not ResourceLoader.exists(background):
		return
	var art := TextureRect.new()
	art.texture = load(background)
	art.set_anchors_preset(Control.PRESET_FULL_RECT)
	art.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	art.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_COVERED
	art.mouse_filter = Control.MOUSE_FILTER_IGNORE
	# Pushed well down so it reads as a setting rather than as a picture with
	# words on it, and so the type never has to fight the art for contrast.
	art.modulate = Color(0.42, 0.40, 0.36, 1.0)
	add_child(art)

	# A vignette out of GradientTexture2D rather than a shader: radial fill is a
	# built-in resource, so it survives export with nothing to compile.
	var grad := Gradient.new()
	grad.set_color(0, Color(shadow, 0.15))
	grad.set_color(1, Color(shadow, 0.95))
	var tex := GradientTexture2D.new()
	tex.gradient = grad
	tex.fill = GradientTexture2D.FILL_RADIAL
	tex.fill_from = Vector2(0.5, 0.5)
	tex.fill_to = Vector2(1.0, 0.5)
	var veil := TextureRect.new()
	veil.texture = tex
	veil.set_anchors_preset(Control.PRESET_FULL_RECT)
	veil.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	veil.stretch_mode = TextureRect.STRETCH_SCALE
	veil.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(veil)


# --- shared look -------------------------------------------------------------

func _rule(width: int) -> ColorRect:
	# A hairline in brass under the title and above the buttons. Cheaper than an
	# ornament texture and it cannot go missing from the export.
	var r := ColorRect.new()
	r.color = Color(gold, 0.55)
	r.custom_minimum_size = Vector2(width, 2)
	r.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	return r


func _heading(text: String, size: int, color: Color) -> Label:
	var l := Label.new()
	l.text = text
	l.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", color)
	return l


func _menu_button(text: String) -> Button:
	var b := Button.new()
	b.text = text
	b.flat = true
	b.focus_mode = Control.FOCUS_ALL
	b.add_theme_font_size_override("font_size", ITEM_SIZE)
	b.add_theme_color_override("font_color", stone)
	b.add_theme_color_override("font_hover_color", gold)
	b.add_theme_color_override("font_focus_color", gold)
	b.add_theme_color_override("font_pressed_color", gold)
	# A flat button in Godot still ships a hover panel; replaced with a wash of
	# the accent so the highlight reads as oxblood rather than editor grey.
	var hover := StyleBoxFlat.new()
	hover.bg_color = Color(accent, 0.35)
	hover.content_margin_left = 24
	hover.content_margin_right = 24
	hover.content_margin_top = 6
	hover.content_margin_bottom = 6
	b.add_theme_stylebox_override("hover", hover)
	b.add_theme_stylebox_override("focus", hover)
	return b


func _column(sep: int) -> VBoxContainer:
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", sep)
	v.alignment = BoxContainer.ALIGNMENT_CENTER
	return v


# --- title screen ------------------------------------------------------------

func _build_main() -> Control:
	var centre := CenterContainer.new()
	centre.set_anchors_preset(Control.PRESET_FULL_RECT)
	var col := _column(10)

	# Letterspacing is what makes a default font read as carved rather than as a
	# UI label; Godot has no tracking property, so the string carries it.
	var spaced := ""
	for i in title_text.to_upper().length():
		spaced += title_text.to_upper()[i] + (" " if i < title_text.length() - 1 else "")
	col.add_child(_heading(spaced, TITLE_SIZE, gold))
	col.add_child(_rule(520))
	var spacer := Control.new()
	spacer.custom_minimum_size = Vector2(0, 40)
	col.add_child(spacer)

	for entry in [["FIGHT", _on_fight], ["SETTINGS", _on_settings], ["QUIT", _on_quit]]:
		var b := _menu_button(entry[0])
		b.pressed.connect(entry[1])
		col.add_child(b)

	centre.add_child(col)
	return centre


func _on_fight() -> void:
	get_tree().change_scene_to_file(world_scene)


func _on_settings() -> void:
	_main.hide()
	_settings.show()


func _on_quit() -> void:
	get_tree().quit()


# --- settings ----------------------------------------------------------------

func _build_settings() -> Control:
	var margin := MarginContainer.new()
	margin.set_anchors_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 60)

	var panel := PanelContainer.new()
	var box := StyleBoxFlat.new()
	box.bg_color = Color(shadow, 0.92)
	box.border_color = Color(gold, 0.45)
	box.set_border_width_all(1)
	for side in ["left", "right", "top", "bottom"]:
		box.set("content_margin_" + side, 28)
	panel.add_theme_stylebox_override("panel", box)

	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	var col := _column(14)
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	col.add_child(_heading("SETTINGS", 44, gold))
	col.add_child(_rule(420))

	col.add_child(_heading("SOUND", ITEM_SIZE, stone))
	for bus in Settings.BUSES:
		col.add_child(_slider_row(bus, 0.0, 1.0, Settings.volume[bus],
			func(v): Settings.set_volume(bus, v)))

	col.add_child(_heading("DISPLAY", ITEM_SIZE, stone))
	col.add_child(_slider_row("Brightness", -1.0, 1.0, Settings.brightness,
		func(v): Settings.set_brightness(v)))

	for group in Settings.BINDABLE:
		col.add_child(_heading(String(group[0]).to_upper(), ITEM_SIZE, stone))
		for action in group[1]:
			col.add_child(_bind_row(action, group[1][action]))

	var row := HBoxContainer.new()
	row.alignment = BoxContainer.ALIGNMENT_CENTER
	row.add_theme_constant_override("separation", 30)
	var reset := _menu_button("RESET KEYS")
	reset.pressed.connect(_on_reset)
	var back := _menu_button("BACK")
	back.pressed.connect(_on_back)
	row.add_child(reset)
	row.add_child(back)
	col.add_child(_rule(420))
	col.add_child(row)

	scroll.add_child(col)
	panel.add_child(scroll)
	margin.add_child(panel)
	return margin


func _row(text: String) -> HBoxContainer:
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 20)
	var l := Label.new()
	l.text = text
	l.custom_minimum_size = Vector2(220, 0)
	l.add_theme_font_size_override("font_size", LABEL_SIZE)
	l.add_theme_color_override("font_color", stone)
	h.add_child(l)
	return h


func _slider_row(text: String, lo: float, hi: float, value: float,
		apply: Callable) -> HBoxContainer:
	var h := _row(text)
	var s := HSlider.new()
	s.min_value = lo
	s.max_value = hi
	s.step = 0.01
	s.value = value
	s.custom_minimum_size = Vector2(320, 0)
	# Godot's stock slider is editor grey and the only part of the screen that
	# still looks like a tool rather than the game.
	var track := StyleBoxFlat.new()
	track.bg_color = Color(stone, 0.20)
	track.set_content_margin_all(3)
	var filled := StyleBoxFlat.new()
	filled.bg_color = Color(gold, 0.85)
	filled.set_content_margin_all(3)
	s.add_theme_stylebox_override("slider", track)
	s.add_theme_stylebox_override("grabber_area", filled)
	s.add_theme_stylebox_override("grabber_area_highlight", filled)
	var readout := Label.new()
	readout.custom_minimum_size = Vector2(60, 0)
	readout.add_theme_font_size_override("font_size", LABEL_SIZE)
	readout.add_theme_color_override("font_color", gold)
	readout.text = "%d%%" % roundi(value * 100)
	s.value_changed.connect(func(v):
		apply.call(v)
		readout.text = "%d%%" % roundi(v * 100))
	h.add_child(s)
	h.add_child(readout)
	return h


func _bind_row(action: String, text: String) -> HBoxContainer:
	var h := _row(text)
	h.set_meta("action", action)
	var b := _menu_button(Settings.label_for(action))
	b.custom_minimum_size = Vector2(180, 0)
	b.pressed.connect(func():
		_capturing = action
		_capture_button = b
		b.text = "PRESS A KEY")
	h.add_child(b)
	return h


func _unhandled_input(event: InputEvent) -> void:
	if _capturing == "" or not (event is InputEventKey) or not event.pressed:
		return
	# Escape cancels rather than binding: a player who opens the rebind by
	# accident needs a way out that does not cost them the key.
	if event.keycode != KEY_ESCAPE:
		Settings.rebind(_capturing, event.physical_keycode)
	_refresh_binds()
	_capturing = ""
	_capture_button = null
	get_viewport().set_input_as_handled()


func _refresh_binds() -> void:
	# Every row is redrawn, not just the one edited: rebinding a key that was
	# already taken clears the other action, and that row has to stop claiming it.
	for h in _settings.find_children("", "HBoxContainer", true, false):
		var meta = h.get_meta("action", null)
		if meta != null:
			for c in h.get_children():
				if c is Button:
					c.text = Settings.label_for(meta)


func _on_reset() -> void:
	Settings.reset_bindings()
	_refresh_binds()


func _on_back() -> void:
	Settings.save_settings()
	_settings.hide()
	_main.show()
