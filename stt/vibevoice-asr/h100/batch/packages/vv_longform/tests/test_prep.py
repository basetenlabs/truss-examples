"""Equivalence tests for the chunked-request audio preparation (VV_PREP_*).

The fast path must send vLLM the same samples, the same chunk boundaries and the same request
bytes as the code it replaced (30673c1). ``_bounds_ref`` and the ``audio.decode`` /
``audio.wav_data_uri`` / ``json.dumps`` path are the references.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from vv_longform import audio, chunking
from vv_longform.config import Settings, load
from vv_longform.server import Proxy

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def _bounds_ref(pcm, sample_rate, chunk_s, search_s):
    """chunking.bounds as of 30673c1 (whole-file float32 copy)."""
    n = int(pcm.shape[0])
    step = int(chunk_s * sample_rate)
    if chunk_s <= 0 or n <= step:
        return [(0, n)]
    frame = max(1, int(chunking.FRAME_S * sample_rate))
    search = int(max(0.0, search_s) * sample_rate)
    x = pcm.astype(np.float32)
    cuts = [0]
    while n - cuts[-1] > step:
        target = cuts[-1] + step
        lo = max(cuts[-1] + frame, target - search)
        hi = min(n - frame, target + search)
        cut = target
        if hi > lo:
            starts = np.arange(lo, hi - frame + 1, frame // 2 or 1)
            if starts.size:
                energy = np.array([float(np.mean(x[s : s + frame] ** 2)) for s in starts])
                cut = int(starts[int(np.argmin(energy))] + frame // 2)
        cut = min(max(cut, cuts[-1] + 1), n)
        cuts.append(cut)
    if len(cuts) > 1 and n - cuts[-1] < step * 0.1:
        cuts.pop()
    cuts.append(n)
    return list(zip(cuts[:-1], cuts[1:]))


def _speechy(seconds: float, rate: int = 24000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    env = (0.5 + 0.5 * np.sin(0.7 * t)) * ((t % 13) > 1.2)
    x = 0.3 * np.sin(2 * np.pi * (200 + 50 * np.sin(0.3 * t)) * t) * env + 0.05 * rng.standard_normal(t.size)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


_NOISY = ("aevalsrc='0.3*sin(2*PI*(200+50*sin(0.3*t))*t)*(0.5+0.5*sin(0.7*t))*gt(mod(t\\,13)\\,1.2)"
          "+0.05*(random(0)-0.5)':s=44100:d={d}")


def _m4a(seconds: float, pns: bool) -> bytes:
    """AAC-LC m4a (moov at the end) of gated tones over low noise; ffmpeg's encoder codes some of
    the noise with PNS unless told not to."""
    fd, path = tempfile.mkstemp(suffix=".m4a")  # mp4 muxing needs a seekable output
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", _NOISY.format(d=seconds),
                        "-c:a", "aac", "-b:a", "96k", "-aac_pns", "1" if pns else "0", "-y", path],
                       check=True)  # fmt: skip
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


@pytest.fixture(scope="module")
def m4a_plain():
    return _m4a(260, pns=False)


@pytest.fixture(scope="module")
def m4a_pns():
    return _m4a(260, pns=True)


def _decode_long(data: bytes, **kw):
    async def go():
        with ThreadPoolExecutor(5) as pool:
            return await audio.decode_long(data, pool, **{"workers": 4, "min_seg_s": 60.0, **kw})

    return asyncio.run(go())


def test_wav_b64_matches_to_wav():
    rng = np.random.default_rng(1)
    for n in (0, 1, 2, 3, 4097, 24000 * 7 + 5):
        pcm = rng.integers(-32768, 32767, n, dtype=np.int16)
        assert audio.wav_b64(pcm) == base64.b64encode(audio.to_wav(pcm))
        assert audio.wav_header(n) == audio.to_wav(pcm)[:44]
    big = rng.integers(-32768, 32767, 24000 * 100, dtype=np.int16)
    a, b = 12345, 24000 * 61 + 7  # a non-contiguous-origin slice, as chunks are
    assert (audio.WAV_URI_PREFIX + audio.wav_b64(big[a:b])).decode() == audio.wav_data_uri(big[a:b])


def test_bounds_match_reference():
    rng = np.random.default_rng(2)
    cases = [(_speechy(1900, seed=3), 24000, 300, 20.0), (_speechy(700, seed=4), 24000, 60, 5.0),
             (rng.integers(-3000, 3000, 24000 * 400, dtype=np.int16), 24000, 120, 20.0),
             (_speechy(95, 16000), 16000, 30, 5.0), (_speechy(50, 16000), 16000, 30, 0.0)]  # fmt: skip
    for pcm, rate, chunk_s, search_s in cases:
        assert chunking.bounds(pcm, rate, chunk_s, search_s) == _bounds_ref(pcm, rate, chunk_s, search_s)


@needs_ffmpeg
def test_parallel_decode_is_sample_identical(m4a_plain):
    ref = asyncio.run(audio.decode(m4a_plain))
    pcm, info = _decode_long(m4a_plain)
    assert (info.mode, info.n_segments, info.seam_mismatch) == ("parallel", 4, 0)
    assert pcm.dtype == ref.dtype and np.array_equal(pcm, ref)


@needs_ffmpeg
def test_seam_mismatch_falls_back_to_sequential(m4a_plain):
    ref = asyncio.run(audio.decode(m4a_plain))
    # Without a pre-roll each segment's first AAC frame has no overlap-add partner: seams differ.
    pcm, info = _decode_long(m4a_plain, preroll_s=0.0)
    assert info.mode == "fallback" and info.seam_mismatch == 3
    assert np.array_equal(pcm, ref)
    loose, info = _decode_long(m4a_plain, preroll_s=0.0, fallback=False)
    assert info.mode == "loose" and loose.shape == ref.shape and not np.array_equal(loose, ref)


@needs_ffmpeg
def test_pns_aac_known_difference(m4a_pns):
    """Documented gap: ffmpeg's AAC decoder draws PNS noise from a generator whose state carries
    across the whole stream, so a segment decode fills noise-substituted bands with a different
    noise realization. The seam check sees this only where PNS is active inside the overlap, so
    VV_PREP_PARALLEL=1 can send slightly different samples for PNS-coded AAC (hence default off)."""
    ref = asyncio.run(audio.decode(m4a_pns))
    pcm, info = _decode_long(m4a_pns)
    assert pcm.shape == ref.shape
    if info.mode == "parallel":  # check passed; any difference is the PNS noise only
        diff = np.flatnonzero(pcm != ref)
        assert diff.size < 1e-3 * ref.size


@needs_ffmpeg
def test_short_or_disabled_decode_is_sequential(m4a_plain):
    ref = asyncio.run(audio.decode(m4a_plain))
    for kw in ({"min_parallel_s": 300.0}, {"parallel": False}, {"min_seg_s": 200.0}):
        pcm, info = _decode_long(m4a_plain, **kw)
        assert (info.mode, info.n_segments) == ("seq", 1) and np.array_equal(pcm, ref)


def test_prep_knobs(monkeypatch):
    for k in ("VV_PREP_PARALLEL", "VV_PREP_THREADS", "VV_PREP_SEAM_FALLBACK", "VV_PREP_MIN_SEG_S"):
        monkeypatch.delenv(k, raising=False)
    s = load()
    assert s.prep_parallel and s.prep_seam_fallback and s.prep_threads == 4 and s.prep_min_seg_s == 120.0
    monkeypatch.setenv("VV_PREP_PARALLEL", "0")
    assert not load().prep_parallel


@needs_ffmpeg
@pytest.mark.parametrize("parallel", [True, False])
def test_chunked_chat_requests_identical_to_reference(m4a_plain, parallel, capsys):
    """Chunk boundaries, WAV payloads and request bytes equal what 30673c1 built."""
    raws: list[bytes] = []

    async def chat(request):
        raws.append(await request.read())
        return web.json_response({"choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": "[]"}}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}})  # fmt: skip

    async def clip(request):
        return web.Response(body=m4a_plain, content_type="audio/mp4")

    body = {"model": "vibevoice", "temperature": 0, "max_tokens": 500, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe this audio."}, {"type": "audio_url", "audio_url": {"url": ""}}]}]}  # fmt: skip

    async def go():
        app = web.Application(client_max_size=1 << 30)
        app.router.add_post("/v1/chat/completions", chat)
        app.router.add_get("/clip.m4a", clip)
        up = TestServer(app)
        await up.start_server()
        u = str(up.make_url("")).rstrip("/")
        s = replace(Settings(), upstream=u, chunk_s=60.0, cut_search_s=5.0, speaker_link="none",
                    prep_min_seg_s=60.0, prep_parallel=parallel, canonical_prompt=False,
                    max_tokens_per_s=None, max_model_len=32768)  # fmt: skip  (30673c1's settings)
        px = TestServer(Proxy(s).app())
        await px.start_server()
        req = copy.deepcopy(body)
        req["messages"][0]["content"][1]["audio_url"]["url"] = u + "/clip.m4a"
        try:
            async with ClientSession() as c:
                async with c.post(str(px.make_url("/v1/chat/completions")), json=req) as r:
                    assert r.status == 200, await r.text()
        finally:
            await px.close()
            await up.close()
    
    asyncio.run(go())
    ref = asyncio.run(audio.decode(m4a_plain))
    cuts = _bounds_ref(ref, audio.SAMPLE_RATE, 60.0, 5.0)
    assert len(raws) == len(cuts) == 5
    # Old body per chunk: the client's body with the chunk's WAV URI and the README max_tokens rule,
    # serialized with json.dumps. Requests run concurrently, so compare as sets.
    want = set()
    for a, b in cuts:
        dur = (b - a) / audio.SAMPLE_RATE
        old = copy.deepcopy(body)
        old["messages"][0]["content"][1]["audio_url"]["url"] = audio.wav_data_uri(ref[a:b])
        old["max_tokens"] = max(1, min(max(256, min(32768, int(20 * dur + 200))),
                                       32768 - int(np.ceil(dur * 7.5)) - 128))  # fmt: skip
        want.add(json.dumps(old).encode())
    assert set(raws) == want
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("timing ")][0]
    assert f"prep_mode={'parallel' if parallel else 'seq'}" in line and "fetch_ms=" in line
