"""Incremental detokenisation for NeMo's streaming RNNT hypotheses (``MT_INC_DETOK``).

Why: ``Hypothesis.merge_`` (rnnt_utils.py) appends the chunk's tokens and sets ``text=None``, so
``RNNTDecoding.decode_hypothesis`` (rnnt_decoding.py) re-runs ``y_sequence.tolist()``, the blank
filter, a per-id ``id_to_piece`` loop and ``SentencePiece.decode_pieces`` over the WHOLE session for
every active speaker on every 1.12 s step, then the space-before-punctuation regex over the whole
string. That is the only per-step cost that grows with session length (FASTSTEP.md s.4).

What: the same string, built from a cached committed prefix plus the new tail. The cache lives on the
``Hypothesis`` object itself (``hyp._mt_inc``), so it follows the object through ``merge_`` (in place)
and through ``deepcopy``. The prefix is only ever cut at a word-start piece (``U+2581`` + letters,
len > 1) whose committed decode is non-empty and ends in a non-space character: there
``decode_pieces(A + B) == decode_pieces(A) + " " + decode_pieces(B)`` and the regex
``(\\s)(punct) -> \\2`` cannot match across the seam, so ``text == text(A) + sub(" " + raw(B))``.
Anything the rule does not cover (TDT / multi-blank models, aggregate or legacy tokenizers, language
tag stripping, blanks inside ``y_sequence``) falls back to NeMo's full decode for that hypothesis.

``MT_INC_DETOK_VERIFY=1`` runs the original full decode next to the incremental one on every step,
counts mismatches, logs the first few and keeps the original text -- the identity proof used in
LONGAUDIO.md; off in the preset.
"""

import logging

logger = logging.getLogger(__name__)

WORD_START = "▁"   # SentencePiece word-boundary marker


class IncrementalDetok:
    """Drop-in for ``RNNTDecoding.decode_hypothesis`` on one decoding instance."""

    def __init__(self, decoding, tokenizer, verify=False):
        self.dec = decoding
        self.orig = decoding.decode_hypothesis      # bound original (full decode)
        self.verify = verify
        self.blank_id = decoding.blank_id
        self.punct = getattr(decoding, "space_before_punct_pattern", None) \
            if getattr(decoding, "supported_punctuation", None) else None
        self.token_set = bool(getattr(decoding, "compute_hypothesis_token_set", False))
        self.stats = {"hyps": 0, "inc": 0, "full": 0, "mismatch": 0, "tail_tokens": 0}
        self._piece = {}                              # id -> piece (str)
        self._ws = {}                                 # id -> is word-start piece
        self.sp = getattr(tokenizer, "tokenizer", None)   # sentencepiece.SentencePieceProcessor
        self.special = getattr(tokenizer, "id_to_special_token", {}) or {}
        self.vocab = getattr(tokenizer, "original_vocab_size", None)
        # Conditions under which the split rule is exact; otherwise every hyp takes the full path.
        reasons = []
        if type(tokenizer).__name__ != "SentencePieceTokenizer" or self.sp is None:
            reasons.append(f"tokenizer={type(tokenizer).__name__}")
        if getattr(tokenizer, "legacy", False):
            reasons.append("legacy tokenizer")
        if getattr(decoding, "_is_tdt", False) or getattr(decoding, "big_blank_durations", None):
            reasons.append("tdt/multi-blank")
        if getattr(decoding, "strip_lang_tags", False):
            reasons.append("strip_lang_tags")
        self.disabled_reason = ", ".join(reasons)
        self.enabled = not reasons
        logger.info("incremental detok: %s (blank_id=%s, punct=%s, token_set=%s, compute_timestamps=%s, verify=%s)",
                    "enabled" if self.enabled else f"DISABLED ({self.disabled_reason})", self.blank_id,
                    bool(self.punct), self.token_set, getattr(decoding, "compute_timestamps", None), verify)

    # ------------------------------------------------------------------ pieces
    def piece(self, tid):
        p = self._piece.get(tid)
        if p is None:
            if self.vocab is not None and tid >= self.vocab:
                p = self.special.get(tid, "")
                ws = False
            else:
                p = self.sp.id_to_piece(tid)
                ws = len(p) > 1 and p[0] == WORD_START
            self._piece[tid] = p
            self._ws[tid] = ws
        return p

    def _sub(self, s):
        return self.punct.sub(r"\2", s) if self.punct is not None else s

    def _raw(self, pieces):
        return self.sp.decode_pieces(pieces)

    # ------------------------------------------------------------------ core
    def _commit_point(self, ids):
        """Largest j >= 1 with ids[j] a word start such that decode(ids[:j]) is non-empty and ends in a
        non-space char (the seam invariant). Returns (j, raw_done) or (0, "")."""
        tries = 0
        for j in range(len(ids) - 1, 0, -1):
            if self._ws.get(ids[j]) is None:
                self.piece(ids[j])
            if self._ws[ids[j]]:
                raw = self._raw([self.piece(t) for t in ids[:j]])
                if raw and not raw[-1].isspace():
                    return j, raw
                tries += 1          # e.g. a lone "▁" piece before this word: try an earlier boundary
                if tries >= 3:
                    break
        return 0, ""

    def _init_state(self, hyp, ids):
        j, raw = self._commit_point(ids)
        if j:
            hyp._mt_inc = [j, raw, self._sub(raw), [self.piece(t) for t in ids[:j]] if self.token_set else None]
        else:
            hyp._mt_inc = [0, "", "", [] if self.token_set else None]

    def decode_one(self, hyp):
        y = hyp.y_sequence
        n = len(y)
        st = getattr(hyp, "_mt_inc", None)
        if st is None or st[0] > n:
            ids = y.tolist() if not isinstance(y, list) else list(y)
            ids = [p for p in ids if p != self.blank_id]
            pieces = [self.piece(t) for t in ids]
            hyp.text = self._sub(self._raw(pieces))
            if self.token_set:
                hyp.tokens = pieces
            self._init_state(hyp, ids)
            self.stats["full"] += 1
            return
        n_done, raw_done, text_done, tok_done = st
        tail = y[n_done:]
        tail = tail.tolist() if not isinstance(tail, list) else list(tail)
        if any(p == self.blank_id for p in tail):
            hyp._mt_inc = None                 # blanks inside the tail: rule not exact -> full path
            return self.decode_one(hyp)
        self.stats["inc"] += 1
        self.stats["tail_tokens"] += len(tail)
        if not tail:
            hyp.text = text_done
            if self.token_set:
                hyp.tokens = list(tok_done)
            return
        pieces = [self.piece(t) for t in tail]
        if n_done and not self._ws[tail[0]]:
            hyp._mt_inc = None                 # invariant broken (cannot happen with append-only y)
            return self.decode_one(hyp)
        raw_tail = self._raw(pieces)
        if n_done:
            hyp.text = text_done + self._sub(" " + raw_tail)
        else:
            hyp.text = self._sub(raw_tail)
        if self.token_set:
            hyp.tokens = tok_done + pieces
        # Advance the committed prefix to the last word start inside the tail.
        j, raw_j = self._commit_point(tail)
        if j:
            if n_done:
                st[1] = raw_done + " " + raw_j
                st[2] = text_done + self._sub(" " + raw_j)
            else:
                st[1] = raw_j
                st[2] = self._sub(raw_j)
            st[0] = n_done + j
            if self.token_set:
                st[3] = tok_done + pieces[:j]

    def decode_hypothesis(self, hypotheses_list):
        if not self.enabled:
            return self.orig(hypotheses_list)
        for hyp in hypotheses_list:
            self.stats["hyps"] += 1
            self.decode_one(hyp)
            if self.verify:
                self._check(hyp)
        return hypotheses_list

    def _check(self, hyp):
        text, tokens = hyp.text, getattr(hyp, "tokens", None)
        st = hyp._mt_inc
        hyp.text = None
        self.orig([hyp])                     # NeMo's full decode
        ok = hyp.text == text and (not self.token_set or hyp.tokens == tokens)
        if not ok:
            self.stats["mismatch"] += 1
            if self.stats["mismatch"] <= 5:
                i = next((k for k in range(min(len(text), len(hyp.text))) if text[k] != hyp.text[k]),
                         min(len(text), len(hyp.text)))
                logger.warning("INCDETOK mismatch #%d at char %d: inc=%r full=%r (n_done=%d, len=%d)",
                               self.stats["mismatch"], i, text[max(0, i - 40): i + 40],
                               hyp.text[max(0, i - 40): i + 40], st[0] if st else -1, len(hyp.y_sequence))
            hyp._mt_inc = None               # resync from the full text next step
        else:
            hyp.text = text                  # identical; keep the incremental object state

    def summary(self):
        s = dict(self.stats)
        s["enabled"] = self.enabled
        return s
