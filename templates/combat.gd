extends Node
class_name CombatController

# Q/E striking with chained follow-ups.
#
# Q is the left hand and E is the right, and what a press produces depends on
# what preceded it: the same Q opens with a jab, follows a jab with a body shot,
# and follows a cross with a left uppercut. That is the whole point of a combo
# system -- the second beat is a different move, not the first one replayed.
#
# The clips themselves are authored per archetype in templates/blender_motion.py
# -- the grappler plants and swings, the striker is bladed and leads with kicks
# -- so the same six buttons play differently on each character.

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

# Timing comes from the clips themselves rather than from a table of frames.
#
# Each character's moveset is authored at its own tempo -- the grappler's hook
# takes 0.75s and the striker's spinning backfist 0.42s -- so the animation
# already knows this character's frame data. Reading it back means the hitbox
# cannot drift from the picture, and a second character does not need a second
# table that somebody has to remember to update.
#
# CONTACT is where in the clip the blow lands, as a fraction of its length. It
# is the same instant the moveset authors as its contact key.
const CONTACT := {
	"jab":            0.52,
	"cross":          0.58,
	"overhand":       0.56,
	"left_uppercut":  0.62,
	"right_uppercut": 0.55,
	"left_bodyshot":  0.58,
}

const ACTIVE_SECONDS := 0.09    # how long the hitbox stays live
const CANCEL_SLACK := 0.04      # chaining opens just after the hitbox shuts

# Damage scales with how long the move takes. That is the genre's own bargain --
# a slow button hurts more -- and it makes the archetype's damage fall out of
# its animation: the grappler's committed swings hit hard because they are slow,
# and the striker's fast buttons do not, with nothing to configure per character.
const DAMAGE_PER_SECOND := 14.0

const FALLBACK_SECONDS := 0.5   # a move whose clip is missing still has timing

var _length: Dictionary = {}    # move -> seconds, read from the loaded clips

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
	for move in CONTACT.keys():
		var seconds := FALLBACK_SECONDS
		if _anim and _anim.has_animation(move):
			var clip := _anim.get_animation(move)
			clip.loop_mode = Animation.LOOP_NONE
			if clip.length > 0.001:
				seconds = clip.length
		_length[move] = seconds


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

	var contact := _contact_seconds(_move)
	if not _did_hit and _elapsed >= contact and _elapsed <= contact + ACTIVE_SECONDS:
		_did_hit = true
		strike_landed.emit(_move, _damage(_move))

	# a buffered press fires the moment the cancel window opens, provided it was
	# not pressed so long ago that the player has plainly changed their mind
	if _buffered != "" and _elapsed >= contact + ACTIVE_SECONDS + CANCEL_SLACK:
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
	return _length.get(move, FALLBACK_SECONDS)


func _contact_seconds(move: String) -> float:
	return _total_seconds(move) * float(CONTACT.get(move, 0.55))


func _damage(move: String) -> float:
	return _total_seconds(move) * DAMAGE_PER_SECOND


func _play(move: String) -> void:
	if _anim == null or not _anim.has_animation(move):
		return
	# Played at its authored speed. The timing above is read back FROM the clip,
	# so there is nothing left to reconcile -- the hitbox opens when the fist
	# arrives because both are measured from the same animation.
	_anim.play(move, 0.06)


## For a third beat later: extend COMBOS with three-hand keys and raise
## MAX_CHAIN. Nothing else here assumes a depth of two.
func combos_for_debug() -> Dictionary:
	return COMBOS
