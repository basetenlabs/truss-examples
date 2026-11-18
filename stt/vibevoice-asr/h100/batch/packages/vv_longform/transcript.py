"""Parse VibeVoice's JSON transcript and stitch chunk transcripts.

VibeVoice-ASR replies with a JSON array such as
``[{"Start": 0.0, "End": 3.2, "Speaker": 0, "Content": "..."}]``; some
prompts produce ``"Start time"`` / ``"End time"`` / ``"Speaker ID"`` keys.
A reply cut by ``max_tokens`` is an unterminated array: keep the complete
objects before the cut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

_KEYS = {
    "Start": "start", "Start time": "start",
    "End": "end", "End time": "end",
    "Speaker": "speaker", "Speaker ID": "speaker",
    "Content": "text",
}  # fmt: skip


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


def _time(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        parts = v.strip().split(":")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(v)
        except ValueError:
            return 0.0
    return 0.0


def _objects(text: str) -> list:
    s = text.strip()
    if "```json" in s:
        a = s.find("```json") + 7
        b = s.find("```", a)
        s = s[a : b if b != -1 else len(s)].strip()
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else [out]
    except (json.JSONDecodeError, ValueError):
        pass
    # Truncated array: decode objects one by one until the first broken one.
    a = s.find("[")
    if a == -1:
        return []
    dec = json.JSONDecoder()
    i, out = a + 1, []
    while i < len(s):
        while i < len(s) and s[i] in " \t\r\n,":
            i += 1
        if i >= len(s) or s[i] == "]":
            break
        try:
            obj, i = dec.raw_decode(s, i)
        except (json.JSONDecodeError, ValueError):
            break
        out.append(obj)
    return out


def parse(text: str | None) -> list[Segment]:
    """Segments of one reply; an unparsable reply yields no segments."""
    if not text:
        return []
    segs = []
    for item in _objects(text):
        if not isinstance(item, dict):
            continue
        row = {_KEYS[k]: v for k, v in item.items() if k in _KEYS}
        if "text" not in row:
            continue
        start = _time(row.get("start", 0.0))
        end = _time(row.get("end", start))
        spk = row.get("speaker")
        segs.append(
            Segment(start, max(end, start), "unknown" if spk in (None, "") else str(spk),
                    str(row["text"]).strip())
        )  # fmt: skip
    return segs


def stitch(
    chunks: list[tuple[float, float, list[Segment]]],
    labels: dict[tuple[int, str], str] | None = None,
) -> list[Segment]:
    """Shift each chunk's segments by its offset and clamp them into the chunk.

    ``labels`` maps (chunk index, local speaker) to a file-wide speaker (``linking.labels``,
    ``VV_LINK=embed``). Without it (``VV_LINK=none``) each chunk's local speaker ``k``
    becomes ``c<i>s<k>``, so a speaker who spans two chunks gets two labels, which
    tcpWER/cpWER count as speaker errors. A single chunk keeps the model's own labels.
    """
    out: list[Segment] = []
    for i, (offset, length, segs) in enumerate(chunks):
        for s in segs:
            start = min(max(s.start, 0.0), length) + offset
            end = min(max(s.end, s.start, 0.0), length) + offset
            spk = s.speaker
            if len(chunks) > 1 and spk != "unknown":
                spk = labels.get((i, spk), f"c{i}s{spk}") if labels is not None else f"c{i}s{spk}"
            out.append(Segment(start, end, spk, s.text))
    return out


def _speaker_out(label: str):
    """An int for a numeric label, as the model itself writes ``"Speaker": 0``; other labels
    (``unknown``, chunk-local ``c<i>s<k>``) stay strings."""
    return int(label) if label.isascii() and label.isdigit() else label


def to_model_json(segs: list[Segment]) -> str:
    """Stitched segments in the model's own reply format (chat completions route)."""
    rows = [
        {"Start": round(s.start, 2), "End": round(s.end, 2), "Speaker": _speaker_out(s.speaker), "Content": s.text}
        for s in segs
    ]
    return json.dumps(rows, ensure_ascii=False)


def to_verbose_json(segs: list[Segment], duration: float) -> dict:
    """OpenAI ``verbose_json`` with a ``speaker`` per segment (b10-bench
    ``openai_transcriptions_tagged`` reads ``speaker``/``start``/``end``/``text``)."""
    rows = []
    for i, s in enumerate(segs):
        spk = s.speaker if s.speaker == "unknown" or not s.speaker.isdigit() else f"S{s.speaker}"
        rows.append({"id": i, "start": round(s.start, 3), "end": round(s.end, 3),
                     "speaker": spk, "text": s.text})  # fmt: skip
    return {
        "task": "transcribe",
        "duration": round(duration, 3),
        "text": " ".join(s.text for s in segs if s.text),
        "segments": rows,
    }
