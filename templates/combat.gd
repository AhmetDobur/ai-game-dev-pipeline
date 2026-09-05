extends Node
class_name CombatController

# Q/E striking with chained follow-ups.
#
# Q is the left hand and E is the right, and what a press produces depends on
# what preceded it: the same Q opens with a jab, follows a jab with a body shot,
# and follows a cross with a left uppercut. That is the whole point of a combo
# system -- the second beat is a different move, not the first one replayed.
#
# The clips themselves are real boxing mocap, cut out of continuous CMU
# shadowboxing by templates/punch_mining.py, so a jab here is somebody's actual
# jab rather than a procedural arm swing.

signal strike_started(move: String, chain: int)
signal strike_landed(move: String, damage: float)

# Sequence of hands -> move. Keys read left to right in the order pressed.
const COMBOS := {
	"q": "jab",
	"e": "cross",
	"qe": "overhand",
	"eq": "left_uppercut",
	"qq": "left_bodyshot",
	"ee": "right_uppercut",
}

# Frame data at 60 fps, the rate fighting games are authored and discussed in.
#
# startup  frames before the hand can hit anything
# active   frames the hitbox is live
# recovery frames after that before neutral
# cancel   frame from which the next input chains, counted from the move's start
#
# cancel sits inside recovery on purpose: chaining is allowed once the hitbox
# closes but before the animation has fully settled, which is what makes a combo
# feel connected instead of like two separate punches queued back to back.
const FRAMES := {
	"jab":            {"startup": 4, "active": 3, "recovery": 8,  "cancel": 9,  "damage": 4.0},
	"cross":          {"startup": 6, "active": 3, "recovery": 12, "cancel": 11, "damage": 7.0},
	"overhand":       {"startup": 9, "active": 4, "recovery": 16, "cancel": 15, "damage": 11.0},
	"left_uppercut":  {"startup": 7, "active": 4, "recovery": 14, "cancel": 13, "damage": 9.0},
	"right_uppercut": {"startup": 7, "active": 4, "recovery": 15, "cancel": 13, "damage": 10.0},
	"left_bodyshot":  {"startup": 6, "active": 3, "recovery": 12, "cancel": 11, "damage": 6.0},
}

const FPS := 60.0
const BUFFER_SECONDS := 0.28   # a press this long before the cancel window still lands
const CHAIN_SECONDS := 0.45    # after this much neutral the chain is forgotten
const MAX_CHAIN := 2           # the third beat lands when its table is written

var _anim: AnimationPlayer = null
var _chain: String = ""        # hands pressed so far this string, e.g. "qe"
var _chain_expires: float = 0.0
var _move: String = ""         # move currently playing, "" when neutral
var _elapsed: float = 0.0      # seconds since the current move started
var _buffered: String = ""     # hand pressed during a move, waiting for the window
var _buffered_at: float = 0.0
var _did_hit: bool = false
var _now: float = 0.0


func setup(anim: AnimationPlayer) -> void:
	_anim = anim
	# Strikes must not loop: a punch that loops is a windmill. Only the
	# locomotion clips cycle, and player.gd has already set those.
	if _anim:
		for move in FRAMES.keys():
			if _anim.has_animation(move):
				_anim.get_animation(move).loop_mode = Animation.LOOP_NONE


func is_striking() -> bool:
	return _move != ""


## True while the character is committed and should not be steered by movement.
func locks_movement() -> bool:
	return _move != "" and _elapsed < _total_seconds(_move)


func press(hand: String) -> void:
	if hand != "q" and hand != "e":
		return
	if _move == "":
		_begin(hand)
	else:
		# buffer it: pressing slightly early is how everybody plays, and
		# dropping those inputs is what makes a combo system feel unresponsive
		_buffered = hand
		_buffered_at = _now


func tick(delta: float) -> void:
	_now += delta
	if _chain != "" and _move == "" and _now > _chain_expires:
		_chain = ""
	if _move == "":
		return
	_elapsed += delta

	var f: Dictionary = FRAMES[_move]
	var frame := _elapsed * FPS
	if not _did_hit and frame >= f.startup and frame <= f.startup + f.active:
		_did_hit = true
		strike_landed.emit(_move, f.damage)

	# a buffered press fires the moment the cancel window opens, provided it was
	# not pressed so long ago that the player has plainly changed their mind
	if _buffered != "" and frame >= f.cancel:
		if _now - _buffered_at <= BUFFER_SECONDS:
			var hand := _buffered
			_buffered = ""
			_begin(hand)
			return
		_buffered = ""

	if _elapsed >= _total_seconds(_move):
		_move = ""
		_elapsed = 0.0
		_did_hit = false
		_chain_expires = _now + CHAIN_SECONDS


func _begin(hand: String) -> void:
	var seq := _chain + hand
	if seq.length() > MAX_CHAIN or not COMBOS.has(seq):
		seq = hand                      # chain broken or unwritten: open fresh
	var move: String = COMBOS.get(seq, "")
	if move == "":
		return
	_chain = seq
	_move = move
	_elapsed = 0.0
	_did_hit = false
	_buffered = ""
	_play(move)
	strike_started.emit(move, seq.length())


func _total_seconds(move: String) -> float:
	var f: Dictionary = FRAMES[move]
	return (f.startup + f.active + f.recovery) / FPS


func _play(move: String) -> void:
	if _anim == null or not _anim.has_animation(move):
		return
	# The mocap clip is however long the performer's punch took; the frame data
	# says how long this punch takes. Scale playback so the hitbox frames and
	# the picture agree -- otherwise the hand lands visibly after the damage.
	var want := _total_seconds(move)
	var have := _anim.get_animation(move).length
	var speed := 1.0
	if have > 0.001 and want > 0.001:
		speed = have / want
	_anim.play(move, 0.06, speed)


## For a third beat later: extend COMBOS with three-hand keys and raise
## MAX_CHAIN. Nothing else here assumes a depth of two.
func combos_for_debug() -> Dictionary:
	return COMBOS
