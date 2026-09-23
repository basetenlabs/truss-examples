"""Turn-level segments and live partials for the multitalker streaming truss (``MT_TURN_SEGMENTS`` / ``MT_PARTIALS``).

Why: NeMo's ``ASRState.update_sessionwise_seglsts_for_parallel`` keeps ONE running sentence per speaker and
appends each step's new text to it; a new sentence starts only when that speaker's own silence exceeds
``sent_break_sec`` (30 s in this preset). Other speakers never break it, so a 6-minute 4-speaker meeting
comes out as ~12 segments of ~100 words, two thirds of them spanning another speaker's turn. Every decoded
word IS in those segments from the step it was emitted (there is no hidden partial) -- appended to a block
that started minutes earlier and sorts far up the list.

What: the same words, cut into turns. ``install`` patches the seglst update at class level to record, per
step, the chunk offset and each speaker's token count / decoded length before the step (a few ints; no
decode work on the step path). ``SessionTurns.pull`` (from ``message()``, off the step path) turns the new
tokens into words -- SentencePiece pieces cached by ``IncrementalDetok``, word times from the RNNT emission
frames (0.08 s) -- and feeds a turn builder. Per speaker, the words concatenate to the same sequence NeMo's
segments carry (``MT_TURNS_VERIFY=1`` checks this at the final against ``ASRState.seglsts``).

Two builders (``MT_TURN_OVERLAP``, handshake ``overlap``):
  1 (default) ``OverlapTurnBuilder`` -- one open tail PER speaker. A tail closes on that speaker's own pause
    > ``pause_s``, on its sentence-final punctuation, and on a hand-over: since its last word another speaker
    has talked for >= ``MT_TURN_HANDOVER_S`` or >= ``MT_TURN_HANDOVER_WORDS`` words. A one- or two-word
    backchannel is not a hand-over: it becomes its own short segment overlapping the running turn. Segments
    may overlap in time and say so (``overlap`` / ``overlaps_with``); ``partial`` lists every open tail.
  0 ``TurnBuilder`` -- one open tail in total; any other speaker's word closes it. Under simultaneous speech
    this alternates one/two-word fragments and shows only the latest speaker's tail.
"""

import bisect
import json
import logging
import os

logger = logging.getLogger(__name__)

# A replace-style partial re-sends every closed turn, so the array is built from cached per-segment
# fragments (``_Seg.frag``) and the message is a string join, not a re-serialisation of ~15k word
# dicts per partial at hour-scale (~5 ms of GIL per partial). The bytes stay exactly ``json.dumps``'s
# default layout (``", "`` / ``": "``): the k6 scenario and other consumers pattern-match on it.
def dumps(obj):
    return json.dumps(obj)

WORD_START = "▁"               # SentencePiece word-boundary marker (also emitted as a lone piece)
FRAME_S = 0.08                 # RNNT emission frame (encoder subsampling 8 x 10 ms)
FRAMES_PER_STEP = 14           # tokens of one step have local frames in [0, 14)
SENT_END = ".?!"
TRAIL = "\"')]}"               # closing quotes/brackets after sentence-final punctuation


def _env_bool(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "off", "no")


class TurnFlags:
    """Env defaults; a session's handshake may override ``turn_segments`` / ``partials`` / ``words`` / ``overlap``."""

    def __init__(self):
        self.turn_segments = _env_bool("MT_TURN_SEGMENTS", "1")
        self.partials = _env_bool("MT_PARTIALS", "1")
        self.words = _env_bool("MT_TURN_WORDS", "1")            # words: [{w,start,end}] per segment
        self.pause_s = float(os.environ.get("MT_TURN_PAUSE_S", "1.2"))
        self.sentence_break = _env_bool("MT_TURN_SENTENCE_BREAK", "1")
        self.overlap = _env_bool("MT_TURN_OVERLAP", "1")         # one open tail per speaker (parallel turns)
        self.handover_s = float(os.environ.get("MT_TURN_HANDOVER_S", "1.0"))
        self.handover_words = int(os.environ.get("MT_TURN_HANDOVER_WORDS", "3"))
        # End-of-turn policy between steps (see ``close_silent``): a tail closes ``vad_s`` after the
        # audio went silent, without waiting for the next 1.12 s step to see the pause. 0 = off.
        self.vad_s = float(os.environ.get("MT_TURN_VAD_S", "0"))
        # Longest a tail stays open (s) before the next step closes it. 0 = unbounded.
        self.max_open_s = float(os.environ.get("MT_TURN_MAX_OPEN_S", "0"))
        # Diarizer end of turn: at a step, close a speaker's tail when its activity has been off for the
        # last ``diar_eot_s`` of the chunk (the per-speaker signal; the mixture is rarely silent). 0 = off.
        self.diar_eot_s = float(os.environ.get("MT_TURN_DIAR_EOT_S", "0"))
        # A closed turn's text is not touched by late punctuation: the "." / "?" RNNT emits one step after
        # the word is dropped instead of glued (scoring-neutral; the normaliser strips it). A late word
        # piece is still glued (dropping it truncated words: "accid", "cater"). Default off = NeMo's text.
        self.freeze_closed = _env_bool("MT_TURN_FREEZE_CLOSED", "0")
        # Emit {"type": "speech", ...} onset frames and an "active" speaker list in every frame.
        self.speech_events = _env_bool("MT_SPEECH_EVENTS", "0")
        self.verify = _env_bool("MT_TURNS_VERIFY", "0")

    def as_dict(self):
        return dict(vars(self))

    # Handshake keys a session may override, with their bounds.
    NUMERIC = {"turn_pause_s": ("pause_s", 0.1, 60.0), "turn_handover_s": ("handover_s", 0.1, 60.0),
               "turn_vad_s": ("vad_s", 0.0, 10.0), "turn_max_open_s": ("max_open_s", 0.0, 600.0),
               "turn_diar_eot_s": ("diar_eot_s", 0.0, 1.12)}

    def override(self, opts):
        """A copy of these flags with a session's handshake overrides applied (bad values raise)."""
        f = TurnFlags.__new__(TurnFlags)
        f.__dict__.update(self.__dict__)
        for k, (attr, lo, hi) in self.NUMERIC.items():
            if opts.get(k) is not None:
                v = float(opts[k])
                if not (lo <= v <= hi):
                    raise ValueError(f"{k} must be in [{lo}, {hi}]")
                setattr(f, attr, v)
        if opts.get("turn_handover_words") is not None:
            f.handover_words = max(1, int(opts["turn_handover_words"]))
        if opts.get("speech_events") is not None:
            f.speech_events = bool(int(opts["speech_events"]))
        if opts.get("turn_freeze_closed") is not None:
            f.freeze_closed = bool(int(opts["turn_freeze_closed"]))
        if opts.get("turn_sentence_break") is not None:
            f.sentence_break = bool(int(opts["turn_sentence_break"]))
        return f


# ------------------------------------------------------------------ step-path patch
_RECORD = {"on": False}


def install(U, on: bool):
    """Record (offset, token counts, decoded lengths) before every seglst update on every ASRState.

    Class-level, like the mt_fast patches: both step paths (lock path via ``update_seglsts`` and the
    xsession tick, which calls the ASRState method directly) go through it. ~1 us per step.
    """
    _RECORD["on"] = on
    cls = U.MultiTalkerInstanceManager.ASRState
    if getattr(cls, "_mt_turns_patched", False):
        return
    orig = cls.update_sessionwise_seglsts_for_parallel

    def update_sessionwise_seglsts_for_parallel(self, offset):
        if _RECORD["on"]:
            ev = self.__dict__.get("_mt_turn_events")
            if ev is None:
                ev = self._mt_turn_events = []
            ev.append((float(offset), list(self._prev_token_counts), list(self._prev_decoded_lengths)))
        return orig(self, offset)

    cls.update_sessionwise_seglsts_for_parallel = update_sessionwise_seglsts_for_parallel
    cls._mt_turns_patched = True


# ------------------------------------------------------------------ turn builders (pure Python)
class _Seg:
    __slots__ = ("speaker", "start", "end", "words", "_text", "_dirty", "_frag", "overlaps", "other_n", "other_t0",
                 "other_t1")

    def __init__(self, speaker, start):
        self.speaker, self.start, self.end = speaker, start, start
        self.words, self._text, self._dirty, self._frag = [], "", True, None
        self.overlaps = None            # overlap mode: speakers whose segments share time with this one
        self.other_n = 0                # overlap mode: other speakers' words since this speaker's last word
        self.other_t0 = self.other_t1 = 0.0

    def touch(self):
        """Words/times changed: text and the cached JSON fragment are stale."""
        self._dirty = True
        self._frag = None

    def text(self):
        if self._dirty:
            self._text = " ".join(w["w"] for w in self.words)
            self._dirty = False
        return self._text

    def as_dict(self, with_words, overlap_fields=False):
        d = {"speaker": self.speaker, "start": round(self.start, 2), "end": round(self.end, 2), "text": self.text()}
        if overlap_fields:
            d["overlap"] = bool(self.overlaps)
            d["overlaps_with"] = sorted(self.overlaps) if self.overlaps else []
        if with_words:
            d["words"] = [{"w": w["w"], "start": round(w["start"], 2), "end": round(w["end"], 2)} for w in self.words]
        return d

    def frag(self, with_words, overlap_fields=False):
        """``dumps(as_dict(...))``, cached until the segment changes (a session renders with fixed flags)."""
        if self._frag is None or self._dirty:
            self._frag = dumps(self.as_dict(with_words, overlap_fields))
        return self._frag


class TurnBuilder:
    """Append-only turn list. Words arrive in non-decreasing start order (steps are 1.12 s windows and
    the caller sorts within a step), so at most ONE segment is ever open: the latest by start."""

    def __init__(self, pause_s=1.2, sentence_break=True, max_open_s=0.0, freeze_closed=False):
        self.pause_s = pause_s
        self.sentence_break = sentence_break
        self.max_open_s = max_open_s
        self.freeze_closed = freeze_closed
        self.n_frozen = 0
        self.closed = []
        self.open = None
        self.last_word = {}            # speaker -> (word dict, segment) for cross-step continuations
        self.n_words = 0
        self.n_vad_closed = 0
        self.n_diar_closed = 0

    def _open_segs(self):
        return [self.open] if self.open is not None else []

    def _stale(self, seg, processed_s):
        """Pause after the tail's last word, or the tail open longer than ``max_open_s``."""
        return (processed_s - seg.end > self.pause_s
                or (self.max_open_s > 0 and seg.words and processed_s - seg.start > self.max_open_s))

    def close_silent(self, silence_start_s, now_s, processed_s, vad_s, tol=0.3):
        """Between steps: the audio has been silent since ``silence_start_s`` (handler-side energy VAD).
        Close every open tail whose words all ended by then (RNNT word ends run late, hence ``tol``),
        once the silence is ``vad_s`` long and the step covering its start has run, so the words spoken
        before the silence are already in the tail. Returns the number of tails closed."""
        if now_s - silence_start_s < vad_s or processed_s < silence_start_s + 0.08:
            return 0
        n = 0
        for seg in list(self._open_segs()):
            if seg.words and seg.end <= silence_start_s + tol:
                self._close_seg(seg)
                n += 1
        self.n_vad_closed += n
        return n

    def close_inactive(self, inactive_s, processed_s, eot_s):
        """At a step: ``inactive_s[speaker]`` = seconds the diarizer has had that speaker OFF at the end
        of the chunk just processed (chunk end = ``processed_s``). Close the speaker's tail when that
        run is >= ``eot_s``. Every word in the tail precedes the chunk end, and the speaker was off
        through it, so no word of the tail can belong to a later utterance; RNNT word-end times run
        late, so they are not compared against the off-run. Returns tails closed."""
        n = 0
        for seg in list(self._open_segs()):
            off = inactive_s.get(seg.speaker, 0.0)
            if seg.words and off >= eot_s:
                self._close_seg(seg)
                n += 1
        self.n_diar_closed += n
        return n

    def _close(self):
        if self.open is not None:
            self.closed.append(self.open)
            self.open = None

    @staticmethod
    def _ends_sentence(w):
        w = w.rstrip(TRAIL)
        return bool(w) and w[-1] in SENT_END

    def add_word(self, speaker, w, start, end):
        seg = self.open
        if seg is not None and (seg.speaker != speaker or start - seg.end > self.pause_s):
            self._close()
            seg = None
        if seg is None:
            seg = self.open = _Seg(speaker, start)
        word = {"w": w, "start": start, "end": end}
        seg.words.append(word)
        seg.end = max(seg.end, end)
        seg.touch()
        self.last_word[speaker] = (word, seg)
        self.n_words += 1
        if self.sentence_break and self._ends_sentence(w):
            self._close()

    def extend_word(self, speaker, suffix, end):
        """A step began mid-word (no word-start piece): glue the suffix onto the speaker's last word."""
        lw = self.last_word.get(speaker)
        if lw is None:
            return False
        word, seg = lw
        if self.freeze_closed and not self._is_open(seg) and not suffix.strip(SENT_END + TRAIL + ",;:-"):
            self.n_frozen += 1              # late punctuation for a closed turn: dropped, the turn stays as emitted
            return True
        # A late word-piece ("accid" + "ent") is glued even into a closed turn: dropping it truncates the word.
        word["w"] += suffix
        # A speaker re-activated much later can emit the closing piece of its last word (e.g. the "." of
        # "Mm-hmm."): glue the text (NeMo's text does), but never stretch the segment past the pause.
        if end - word["end"] <= self.pause_s:
            word["end"] = max(word["end"], end)
            seg.end = max(seg.end, end)
        seg.touch()
        if self.sentence_break and self._is_open(seg) and self._ends_sentence(suffix):
            self._close_seg(seg)
        return True

    def _is_open(self, seg):
        return self.open is seg

    def _close_seg(self, seg):
        self._close()

    def advance(self, processed_s):
        """Silence after the open tail longer than the pause (or the tail past ``max_open_s``): close it."""
        if self.open is not None and self._stale(self.open, processed_s):
            self._close()

    def finalize(self):
        self._close()

    def _all(self, include_open):
        return self.closed if not include_open or self.open is None else self.closed + [self.open]

    def segments(self, with_words, include_open=False):
        return [s.as_dict(with_words) for s in self._all(include_open) if s.words]

    def segments_json(self, with_words, include_open=False):
        """The ``segments`` array as JSON text (same content as ``dumps(segments(...))``)."""
        return "[" + ", ".join(s.frag(with_words) for s in self._all(include_open) if s.words) + "]"

    def partial(self):
        s = self.open
        if s is None or not s.words:
            return []
        return [{"speaker": s.speaker, "text": s.text(), "start": round(s.start, 2), "end": round(s.end, 2)}]

    def speakers(self):
        out = {s.speaker for s in self.closed}
        if self.open is not None:
            out.add(self.open.speaker)
        return out

    def words_by_speaker(self):
        out = {}
        for s in self.closed + ([self.open] if self.open is not None else []):
            out.setdefault(s.speaker, []).extend(w["w"] for w in s.words)
        return out


class OverlapTurnBuilder(TurnBuilder):
    """One open tail PER speaker, so simultaneous speech becomes parallel turns instead of word-level
    alternation. A speaker's tail closes on its own pause > ``pause_s`` (seen on ``advance`` or when its
    next word arrives), on its sentence-final punctuation, and on a hand-over: since its last word other
    speakers have talked for >= ``handover_s`` or >= ``handover_words`` words (the floor moved). A one- or
    two-word backchannel is below both thresholds, so it never splits the running turn; it closes on its
    own pause as a short segment overlapping it. ``closed`` is kept sorted by start; when a segment closes,
    it and every other speaker's segment it shares time with are marked ``overlap`` on both sides."""

    def __init__(self, pause_s=1.2, sentence_break=True, handover_s=1.0, handover_words=3, max_open_s=0.0,
                 freeze_closed=False):
        super().__init__(pause_s, sentence_break, max_open_s, freeze_closed)
        self.handover_s = handover_s
        self.handover_words = handover_words
        self.open = {}                 # speaker -> open _Seg
        self.by_speaker = {}           # speaker -> its closed segments in time order
        self._starts = []              # start of each self.closed entry (bisect key)

    def _open_segs(self):
        return list(self.open.values())

    @staticmethod
    def _touch(a, b):
        """Any word of ``a`` inside ``b``'s time range, or vice versa."""
        return (any(w["start"] < b.end and b.start < w["end"] for w in a.words)
                or any(w["start"] < a.end and a.start < w["end"] for w in b.words))

    def _pair(self, a, b):
        if a.words and b.words and self._touch(a, b):
            a.overlaps = a.overlaps or set()
            b.overlaps = b.overlaps or set()
            a.overlaps.add(b.speaker)
            b.overlaps.add(a.speaker)
            a._frag = b._frag = None                         # overlap fields changed; text did not

    def _mark_overlap(self, seg):
        for other in self.open.values():                     # seg itself is already out of self.open
            self._pair(seg, other)
        for spk, segs in self.by_speaker.items():
            if spk == seg.speaker:
                continue
            for other in reversed(segs):                     # a speaker's own segments are time-ordered
                if other.end <= seg.start:
                    break
                self._pair(seg, other)

    def _is_open(self, seg):
        return self.open.get(seg.speaker) is seg

    def _close_seg(self, seg):
        del self.open[seg.speaker]
        self._mark_overlap(seg)
        i = bisect.bisect_right(self._starts, seg.start)
        self._starts.insert(i, seg.start)
        self.closed.insert(i, seg)
        self.by_speaker.setdefault(seg.speaker, []).append(seg)

    def add_word(self, speaker, w, start, end):
        seg = self.open.get(speaker)
        if seg is not None and start - seg.end > self.pause_s:
            self._close_seg(seg)
            seg = None
        if seg is None:
            seg = self.open[speaker] = _Seg(speaker, start)
        else:
            seg.other_n = 0                                  # this speaker talks on: nothing was handed over
        word = {"w": w, "start": start, "end": end}
        seg.words.append(word)
        seg.end = max(seg.end, end)
        seg.touch()
        self.last_word[speaker] = (word, seg)
        self.n_words += 1
        if self.sentence_break and self._ends_sentence(w):
            self._close_seg(seg)
        # Hand-over check on every other speaker's open tail.
        for other in list(self.open.values()):
            if other.speaker == speaker:
                continue
            if other.other_n == 0:
                other.other_t0, other.other_t1 = start, end
            other.other_n += 1
            other.other_t1 = max(other.other_t1, end)
            if other.other_n >= self.handover_words or other.other_t1 - other.other_t0 >= self.handover_s:
                self._close_seg(other)

    def advance(self, processed_s):
        for seg in list(self.open.values()):
            if self._stale(seg, processed_s):
                self._close_seg(seg)

    def finalize(self):
        for seg in sorted(self.open.values(), key=lambda s: (s.start, s.speaker)):
            self._close_seg(seg)

    def _open_sorted(self):
        return sorted((s for s in self.open.values() if s.words), key=lambda s: (s.start, s.speaker))

    def _all(self, include_open):
        segs = self.closed
        if include_open and self.open:
            segs = sorted(self.closed + self._open_sorted(), key=lambda s: s.start)
        return segs

    def segments(self, with_words, include_open=False):
        return [s.as_dict(with_words, overlap_fields=True) for s in self._all(include_open) if s.words]

    def segments_json(self, with_words, include_open=False):
        return "[" + ", ".join(s.frag(with_words, True) for s in self._all(include_open) if s.words) + "]"

    def partial(self):
        return [{"speaker": s.speaker, "text": s.text(), "start": round(s.start, 2), "end": round(s.end, 2)}
                for s in self._open_sorted()]

    def speakers(self):
        return set(self.by_speaker) | set(self.open)

    def words_by_speaker(self):
        out = {}
        for spk in set(self.by_speaker) | set(self.open):
            segs = list(self.by_speaker.get(spk, ()))
            if spk in self.open:
                segs.append(self.open[spk])
            out[spk] = [w["w"] for s in segs for w in s.words]
        return out


# ------------------------------------------------------------------ per-session glue
def _spk(x):
    x = str(x)
    return x if x.startswith("speaker") else f"speaker_{x}"


def _as_list(t, a, b):
    if isinstance(t, dict):                      # compute_timestamps=True packs a dict; we never enable it
        t = t.get("timestep", [])
    t = t[a:b]
    return t.tolist() if hasattr(t, "tolist") else list(t)


class EnergyVAD:
    """Handler-side speech/silence tracker on the raw PCM a connection appends (CPU, ~1 us per frame).

    A frame is speech when its RMS clears both an absolute floor (-46 dBFS) and 4x the running noise
    floor (a slow-rising minimum). Tracks the end of the last speech frame (``silence_start_s``) and
    reports onsets after >= ``gap_s`` of silence. Used for end-of-turn between steps and the optional
    ``speech`` onset frames; never touches the models."""

    ABS = 0.005
    RATIO = 4.0

    def __init__(self, sr=16000, gap_s=0.3):
        self.sr, self.gap_s = sr, gap_s
        self.total = 0
        self.floor = 0.02
        self.in_speech = False
        self.silence_start_s = 0.0         # audio time the current silence began (last speech frame end)
        self.onsets = 0

    @property
    def now_s(self):
        return self.total / self.sr

    def update(self, samples):
        """Feed one appended frame (float32). Returns True on a speech onset after a gap."""
        n = len(samples)
        if n == 0:
            return False
        rms = float((samples.astype("float32") ** 2).mean()) ** 0.5
        self.floor = min(self.floor * 1.01 + 1e-5, rms) if rms > 0 else self.floor
        self.total += n
        speech = rms > max(self.ABS, self.RATIO * self.floor)
        onset = False
        if speech:
            if not self.in_speech:
                onset = self.now_s - self.silence_start_s >= self.gap_s or self.onsets == 0
                self.onsets += int(onset)
            self.in_speech = True
            self.silence_start_s = self.now_s
        else:
            self.in_speech = False
        return onset


class SessionTurns:
    """Owns one turn builder; converts the recorded step events of an ASRState into words."""

    def __init__(self, inc, flags: TurnFlags, words=None, overlap=None):
        self.inc = inc if (inc is not None and getattr(inc, "sp", None) is not None) else None
        self.flags = flags
        self.with_words = flags.words if words is None else bool(words)
        if flags.overlap if overlap is None else bool(overlap):
            self.tb = OverlapTurnBuilder(flags.pause_s, flags.sentence_break, flags.handover_s, flags.handover_words,
                                         flags.max_open_s, flags.freeze_closed)
        else:
            self.tb = TurnBuilder(flags.pause_s, flags.sentence_break, flags.max_open_s, flags.freeze_closed)
        self.text_seen = {}          # fallback path: speaker -> chars of hyp.text consumed
        self.trailing_ws = {}        # speaker -> last step ended on a lone "▁" (next piece starts a word)
        self.stats = {"steps": 0, "words": 0, "ext": 0, "fallback": 0, "clamped": 0, "split": 0}

    # -- tokens -> words --------------------------------------------------------------------------
    def _starts_word(self, tid):
        """A word begins at a ``▁…`` piece OR at a lone ``▁`` (SentencePiece emits the bare marker before
        rare words: ``▁good ▁ ideas`` decodes to "good ideas"; grouping it with the previous word gave a
        word carrying an inner space)."""
        inc = self.inc
        return inc._ws[tid] or inc.piece(tid) == WORD_START

    def _groups(self, ids, frames, pending_ws=False):
        """Split one step's new tokens into (is_continuation, text, start_frame, end_frame) groups.

        ``pending_ws``: the previous step of this speaker ended on a lone ``▁``, so the first piece here
        starts a new word even without its own marker. Returns ``(groups, trailing_ws)``."""
        inc = self.inc
        out, cur, cur_f = [], [], []
        for tid, f in zip(ids, frames):
            inc.piece(tid)
            if self._starts_word(tid) and cur:
                out.append((cur, cur_f))
                cur, cur_f = [], []
            cur.append(tid)
            cur_f.append(f)
        if cur:
            out.append((cur, cur_f))
        groups, trailing_ws = [], False
        for k, (g, gf) in enumerate(out):
            pieces = [inc.piece(t) for t in g]
            raw = inc._raw(pieces)
            if not raw.strip():
                # A lone marker with nothing after it in this step: the next step's first piece is a
                # word start, not a continuation (NeMo's text has the space).
                trailing_ws = k == len(out) - 1 and pieces == [WORD_START]
                continue
            if k == 0 and not self._starts_word(g[0]) and not pending_ws:
                groups.append((True, inc._sub(raw), min(gf), max(gf)))          # mid-word: continuation
                continue
            s = inc._sub(" " + raw)               # NeMo's space-before-punctuation rule at the seam
            if s.startswith(" "):
                groups.append((False, s[1:], min(gf), max(gf)))
            elif groups:
                p = groups[-1]                    # "▁," etc: glue to the previous group of this step
                groups[-1] = (p[0], p[1] + s, p[2], max(p[3], max(gf)))
            else:
                groups.append((True, s, min(gf), max(gf)))   # step starts with punctuation: glue to last word
        return groups, trailing_ws

    def _items_for(self, state, spk, n0, n1, dl0, offset):
        hyp = state.previous_hypothesis[spk]
        if hyp is None or n1 <= n0:
            return []
        speaker = _spk(spk)
        items = []
        if self.inc is not None:
            ids = _as_list(hyp.y_sequence, n0, n1)
            frames = _as_list(hyp.timestamp, n0, n1)
            if len(frames) != len(ids):
                frames = (frames + [frames[-1] if frames else 0] * len(ids))[: len(ids)]
            loc = []
            for f in frames:
                lf = int(f) - int(dl0)
                if lf < 0 or lf >= FRAMES_PER_STEP:
                    self.stats["clamped"] += 1
                    lf = min(max(lf, 0), FRAMES_PER_STEP - 1)
                loc.append(lf)
            groups, self.trailing_ws[spk] = self._groups(ids, loc, self.trailing_ws.get(spk, False))
            for cont, text, f0, f1 in groups:
                # A group is one word by construction; if a decode still yields whitespace (a special token,
                # an unforeseen piece pattern) split it so no ``words`` entry carries a space. Text per
                # speaker is unchanged: segments join words with single spaces.
                parts = text.split()
                if not parts:
                    continue
                if len(parts) > 1:
                    self.stats["split"] += 1
                start, end = offset + f0 * FRAME_S, offset + (f1 + 1) * FRAME_S
                for j, w in enumerate(parts):
                    items.append((start, end, speaker, cont and j == 0, w))
        else:                                     # non-SentencePiece tokenizer: text diff, step-level times
            self.stats["fallback"] += 1
            text = hyp.text or ""
            seen = self.text_seen.get(spk, 0)
            new = text[seen:] if text.startswith(text[:seen]) else text
            self.text_seen[spk] = len(text)
            cont = bool(seen) and bool(new) and not new[0].isspace() and not text[:seen].endswith(" ")
            for i, w in enumerate(new.split()):
                items.append((offset, offset + FRAMES_PER_STEP * FRAME_S, speaker, cont and i == 0, w))
        return items

    # -- public -----------------------------------------------------------------------------------
    def pull(self, state, processed_s, final):
        """Consume the step events recorded on ``state`` since the last call and update the turns."""
        events = state.__dict__.get("_mt_turn_events")
        if events:
            state._mt_turn_events = []
            speakers = list(state.get_speakers())
            for k, (offset, n0s, dl0s) in enumerate(events):
                nxt = events[k + 1][1] if k + 1 < len(events) else None
                items = []
                for spk in speakers:
                    if spk >= len(n0s):
                        continue
                    hyp = state.previous_hypothesis[spk]
                    if hyp is None:
                        continue
                    if nxt is not None and spk < len(nxt):
                        n1 = nxt[spk]
                    else:
                        t = hyp.timestamp
                        n1 = len(t.get("timestep", [])) if isinstance(t, dict) else len(t)
                    items.extend(self._items_for(state, spk, n0s[spk], n1, dl0s[spk], offset))
                items.sort(key=lambda it: (it[0], it[2]))
                for start, end, speaker, cont, text in items:
                    if cont and self.tb.extend_word(speaker, text, end):
                        self.stats["ext"] += 1
                    else:
                        self.tb.add_word(speaker, text, start, end)
                        self.stats["words"] += 1
                self.stats["steps"] += 1
        if final:
            self.tb.finalize()
        else:
            self.tb.advance(processed_s)

    def close_inactive(self, inactive_frames, processed_s):
        """Diarizer end of turn at a step (``flags.diar_eot_s`` > 0): ``inactive_frames[k]`` = trailing
        OFF frames (0.08 s) of speaker k in the chunk that ended at ``processed_s``."""
        if self.flags.diar_eot_s <= 0 or not inactive_frames:
            return 0
        return self.tb.close_inactive({_spk(k): f * FRAME_S for k, f in enumerate(inactive_frames)}, processed_s,
                                      self.flags.diar_eot_s)

    def close_silent(self, silence_start_s, now_s, processed_s):
        """Between-step end of turn from the handler's VAD (``flags.vad_s`` > 0). Tails closed here are
        promoted exactly as a pause seen at the next step would promote them; nothing is revised."""
        if self.flags.vad_s <= 0:
            return 0
        return self.tb.close_silent(silence_start_s, now_s, processed_s, self.flags.vad_s)

    def segments(self, final):
        return self.tb.segments(self.with_words, include_open=final)

    def segments_json(self, final):
        return self.tb.segments_json(self.with_words, include_open=final)

    def partial(self):
        return self.tb.partial()

    def num_speakers(self):
        return len(self.tb.speakers())

    def verify(self, state, sid):
        """Identity check at the final: per speaker, our words == NeMo's seglst words (case-folded)."""
        ours = {k: " ".join(v).lower().split() for k, v in self.tb.words_by_speaker().items()}
        theirs = {}
        for seg in state.seglsts:
            theirs.setdefault(_spk(seg["speaker"]), []).extend(str(seg.get("words", "")).lower().split())
        bad = [k for k in set(ours) | set(theirs) if ours.get(k, []) != theirs.get(k, [])]
        for k in bad[:3]:
            a, b = ours.get(k, []), theirs.get(k, [])
            i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            logger.warning("TURNS mismatch %s %s at word %d: ours=%s nemo=%s (len %d/%d)", sid, k, i,
                           a[max(0, i - 5): i + 5], b[max(0, i - 5): i + 5], len(a), len(b))
        logger.info("TURNS verify %s: speakers=%d identical=%d words=%d segments=%d stats=%s", sid, len(theirs),
                    len(theirs) - len(bad), sum(len(v) for v in ours.values()), len(self.tb.closed), self.stats)
        return not bad
