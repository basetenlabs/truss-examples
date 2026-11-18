"""CPU tests for cross-chunk speaker linking (VV_LINK=embed).

Synthetic embeddings exercise the clustering; a fake embedder runs the proxy end to end.
Two optional tests need torch: the Kaldi fbank check (also needs torchaudio) and a smoke
on the real WeSpeaker checkpoint, which runs only when ``VV_TEST_WESPEAKER_PATH`` points at
a directory holding ``avg_model`` (and ``VV_TEST_SPEECH_WAVS`` lists >= 2 speech WAVs of
different speakers, comma-separated, for the similarity check).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from dataclasses import replace

import numpy as np
import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestServer

from vv_longform import audio, linking, transcript
from vv_longform.config import Settings, load
from vv_longform.server import Proxy
from vv_longform.transcript import Segment

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def noisy(base, rng, sigma):
    return unit(base + rng.normal(0, sigma, base.shape))


# --- clustering ---------------------------------------------------------------------------------


def test_cluster_recovers_speakers_across_chunks():
    rng = np.random.default_rng(0)
    true = [unit(rng.normal(size=256)) for _ in range(4)]
    items, truth = [], []
    for c in range(8):  # each chunk has 2-3 of the 4 speakers, with shuffled local labels
        present = rng.choice(4, size=int(rng.integers(2, 4)), replace=False)
        for local, spk in enumerate(present):
            items.append(linking.Item(c, str(local), c * 300.0 + local, 20.0, noisy(true[spk], rng, 0.04)))
            truth.append(int(spk))
    got = linking.cluster(items, threshold=0.4, min_s=2.0)
    # same true speaker <=> same cluster
    for i in range(len(items)):
        for j in range(len(items)):
            assert (got[i] == got[j]) == (truth[i] == truth[j]), (i, j)
    assert sorted(set(got)) == list(range(4))


def test_cannot_link_inside_a_chunk():
    e = unit(np.ones(8))
    items = [linking.Item(0, "0", 0.0, 10, e), linking.Item(0, "1", 1.0, 10, e), linking.Item(1, "0", 300.0, 10, e)]
    got = linking.cluster(items, threshold=0.4, min_s=2.0)
    assert got[0] != got[1]
    assert got[2] in (got[0], got[1])


def test_threshold_keeps_distinct_speakers_apart():
    a, b = unit([1, 0, 0, 0]), unit([0, 1, 0, 0])
    items = [linking.Item(0, "0", 0.0, 10, a), linking.Item(1, "0", 300.0, 10, b)]
    assert linking.cluster(items, 0.4, 2.0) == [0, 1]
    assert linking.cluster(items, -1.0, 2.0) == [0, 0]  # threshold below every similarity: merge


def test_short_item_joins_nearest_compatible_cluster():
    a, b = unit([1, 0.1, 0, 0]), unit([0, 1, 0.1, 0])
    items = [
        linking.Item(0, "0", 0.0, 20, a), linking.Item(0, "1", 5.0, 20, b),
        linking.Item(1, "0", 300.0, 0.8, unit([0.2, 1, 0, 0])),  # short, closest to b
        linking.Item(1, "1", 301.0, 0.5, None),  # no embedding: own label
    ]  # fmt: skip
    got = linking.cluster(items, 0.9, 2.0)
    assert got[:3] == [0, 1, 1] and got[3] == 2


def test_labels_renumber_by_first_appearance():
    a, b = unit([1, 0]), unit([0, 1])
    items = [linking.Item(0, "1", 0.0, 10, b), linking.Item(0, "0", 4.0, 10, a),
             linking.Item(1, "0", 300.0, 10, b), linking.Item(1, "1", 302.0, 10, a)]  # fmt: skip
    assert linking.labels(items, 0.4, 2.0) == {(0, "1"): "0", (0, "0"): "1", (1, "0"): "0", (1, "1"): "1"}


# --- audio selection ----------------------------------------------------------------------------


def test_speaker_ranges_trim_drop_overlap_and_cap():
    segs = [Segment(0, 10, "0", "a"), Segment(8, 12, "1", "b"), Segment(12, 50, "0", "c"),
            Segment(50, 50.2, "1", "d"), Segment(51, 52, "unknown", "e")]  # fmt: skip
    r = linking.speaker_ranges(segs, length=60.0, max_s=30.0)
    assert set(r) == {"0", "1"}
    total0 = sum(b - a for a, b in r["0"])
    assert total0 == pytest.approx(30.0)
    assert all(not (a < 12 and b > 8) for a, b in r["0"])  # overlap with speaker 1 removed
    assert r["1"] and all(b > a for a, b in r["1"])
    pcm = np.arange(60 * 100, dtype=np.int16)
    assert linking.gather(pcm, 100, 0, [(1.0, 2.0), (3.0, 3.5)]).tolist() == list(range(100, 200)) + list(range(300, 350))


def test_stitch_uses_labels_and_keeps_stub_without():
    parts = [(0.0, 300.0, [Segment(1, 2, "0", "x")]), (300.0, 300.0, [Segment(1, 2, "1", "y"), Segment(3, 4, "unknown", "z")])]
    assert [s.speaker for s in transcript.stitch(parts)] == ["c0s0", "c1s1", "unknown"]
    assert [s.speaker for s in transcript.stitch(parts, {(0, "0"): "0", (1, "1"): "0"})] == ["0", "0", "unknown"]
    assert [s.speaker for s in transcript.stitch(parts[:1], {(0, "0"): "7"})] == ["0"]  # single chunk untouched


def test_link_env(monkeypatch):
    for k in ("VV_LINK", "VV_LINK_THRESHOLD", "VV_CHUNK_S"):
        monkeypatch.delenv(k, raising=False)
    s = load()
    assert s.speaker_link == "embed" and s.link_threshold == 0.30 and s.link_embed  # shipped: chunk 300 s
    monkeypatch.setenv("VV_CHUNK_S", "0")
    assert not load().link_embed  # chunking off
    monkeypatch.setenv("VV_CHUNK_S", "300")
    monkeypatch.setenv("VV_LINK", "none")
    assert not load().link_embed
    monkeypatch.setenv("VV_LINK", "embed")
    monkeypatch.setenv("VV_LINK_THRESHOLD", "0.6")
    assert load().link_embed and load().link_threshold == 0.6
    monkeypatch.setenv("VV_LINK", "overlap")
    with pytest.raises(ValueError):
        load()


# --- proxy end to end with a fake embedder ------------------------------------------------------

FREQS = (200.0, 450.0, 900.0)  # one sine per "speaker"


def voices(seconds_per_turn: float, turns: list[int], rate: int = 24000) -> np.ndarray:
    n = int(seconds_per_turn * rate)
    t = np.arange(n) / rate
    return np.concatenate([(0.3 * np.sin(2 * np.pi * FREQS[s] * t) * 32767).astype(np.int16) for s in turns])


class FreqEmbedder:
    """Embeds a clip as the one-hot of its dominant frequency among FREQS (24 kHz input)."""

    def __init__(self):
        self.calls = 0

    def embed_pcm24k(self, pcm):
        self.calls += 1
        spec = np.abs(np.fft.rfft(pcm.astype(np.float32)))
        freqs = np.fft.rfftfreq(pcm.size, 1 / 24000)
        power = [spec[np.abs(freqs - f) < 5].sum() for f in FREQS]
        v = np.zeros(len(FREQS), dtype=np.float32)
        v[int(np.argmax(power))] = 1.0
        return v


class TurnVLLM:
    """Replies with one segment per 10 s turn, numbering speakers from 0 in order of
    appearance inside each request (like VibeVoice does per chunk)."""

    def __init__(self):
        self.n = 0

    def app(self):
        app = web.Application(client_max_size=1 << 30)
        app.router.add_get("/v1/models", lambda r: web.json_response({"data": [{"id": "vibevoice"}]}))
        app.router.add_post("/v1/chat/completions", self.chat)
        return app

    async def chat(self, request):
        self.n += 1
        body = await request.json()
        url = next(p["audio_url"]["url"] for m in body["messages"] if isinstance(m.get("content"), list)
                   for p in m["content"] if p.get("type") == "audio_url")  # fmt: skip
        _, data = audio.split_data_uri(url)
        import io
        import wave

        with wave.open(io.BytesIO(data)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            rate = w.getframerate()
        rows, local = [], {}
        for k in range(int(round(pcm.size / rate / 10))):
            clip = pcm[k * 10 * rate:(k + 1) * 10 * rate]
            spk = int(np.argmax(FreqEmbedder().embed_pcm24k(clip)))
            local.setdefault(spk, len(local))
            rows.append({"Start": k * 10.0, "End": k * 10.0 + 10.0, "Speaker": local[spk], "Content": f"f{spk}"})
        return web.json_response({"id": "x", "object": "chat.completion", "model": "vibevoice",
                                  "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": json.dumps(rows)}}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}})  # fmt: skip


async def _serve(settings_kw, fn, embedder):
    fake = TurnVLLM()
    up = TestServer(fake.app())
    await up.start_server()
    s = replace(Settings(), upstream=str(up.make_url("")).rstrip("/"),
                **{"canonical_prompt": False, "max_tokens_per_s": None, "prep_parallel": False, **settings_kw})
    proxy = Proxy(s, embedder=embedder)
    srv = TestServer(proxy.app())
    await srv.start_server()
    import aiohttp

    try:
        async with aiohttp.ClientSession() as client:
            return await fn(client, str(srv.make_url("")).rstrip("/"), proxy)
    finally:
        await srv.close()
        await up.close()


@needs_ffmpeg
def test_transcriptions_linked_end_to_end(capsys):
    # 12 turns of 10 s: chunks of 40 s each see the speakers in a different order.
    turns = [0, 1, 0, 1, 2, 1, 2, 0, 1, 0, 2, 2]
    pcm = voices(10.0, turns)
    flac = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "-",
                           "-f", "flac", "-"], input=pcm.tobytes(), capture_output=True, check=True).stdout  # fmt: skip
    emb = FreqEmbedder()
    kw = {"chunk_s": 40.0, "cut_search_s": 0.0, "speaker_link": "embed", "link_min_s": 1.0}

    async def fn(client, base, proxy):
        form = FormData()
        form.add_field("response_format", "verbose_json")
        form.add_field("file", flac, filename="a.flac", content_type="audio/flac")
        async with client.post(base + "/v1/audio/transcriptions", data=form) as r:
            assert r.status == 200, await r.text()
            return await r.json()

    out = asyncio.run(_serve(kw, fn, emb))
    segs = out["segments"]
    assert len(segs) == 12
    # one label per true speaker across chunks
    by_true: dict[str, set] = {}
    for s in segs:
        by_true.setdefault(s["text"], set()).add(s["speaker"])
    assert all(len(v) == 1 for v in by_true.values()), by_true
    assert sorted(x for v in by_true.values() for x in v) == ["S0", "S1", "S2"]
    assert segs[0]["speaker"] == "S0"
    assert emb.calls > 0
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("timing ")][0]
    assert "n_chunks=3" in line and "link=embed" in line and "n_speakers=3" in line


def test_readiness_waits_for_linker():
    async def fn(client, base, proxy):
        async with client.get(base + "/v1/models") as r:
            assert r.status == 503
        proxy.link_ready.set()
        async with client.get(base + "/v1/models") as r:
            assert r.status == 200

    async def go():
        orig = Proxy._load_linker

        async def never(self):
            await asyncio.sleep(3600)

        Proxy._load_linker = never
        try:
            await _serve({"chunk_s": 300.0, "speaker_link": "embed"}, fn, None)
        finally:
            Proxy._load_linker = orig

    asyncio.run(go())


def test_no_linker_when_chunking_off():
    async def fn(client, base, proxy):
        assert proxy._link_task is None and proxy._link_pool is None
        async with client.get(base + "/v1/models") as r:
            assert r.status == 200

    asyncio.run(_serve({"chunk_s": 0.0, "speaker_link": "embed"}, fn, None))


# --- torch-backed checks (optional) -------------------------------------------------------------


def test_fbank_matches_torchaudio():
    torch = pytest.importorskip("torch")
    kaldi = pytest.importorskip("torchaudio.compliance.kaldi")
    from vv_longform import embed

    x = torch.from_numpy(np.random.default_rng(1).normal(0, 3000, 16000 * 3).astype(np.float32))
    ref = kaldi.fbank(x.unsqueeze(0), num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                      sample_frequency=16000, window_type="hamming", use_energy=False)  # fmt: skip
    got = embed.fbank(x, torch.from_numpy(embed.kaldi_mel_banks()))
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-4), (got - ref).abs().max()


class _ToRecorder:
    """Stands in for the ResNet / mel banks: records ``.to()`` targets instead of moving."""

    def __init__(self, seen):
        self.seen = seen

    def to(self, dev):
        self.seen.append(dev)
        return self


@pytest.mark.parametrize("want", ["auto", "cuda", "cuda:0", " CUDA "])
def test_to_device_cuda_path_uses_indexed_device(monkeypatch, want):
    """GPU path without a GPU: link_device=auto must reach set_per_process_memory_fraction
    and .to() as an indexed CUDA device (exec 5 crash-looped on un-indexed ``cuda``)."""
    torch = pytest.importorskip("torch")
    from torch.cuda._utils import _get_device_index
    from vv_longform import embed

    # The stub runs torch's own index check, so it fails exactly like the real call did.
    with pytest.raises(ValueError, match="specified index"):
        _get_device_index(torch.device("cuda"))

    calls = []

    def fake_fraction(fraction, device=None):
        assert isinstance(fraction, float) and 0 < fraction < 1
        calls.append(_get_device_index(device))  # raises on torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", fake_fraction)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)

    e = object.__new__(embed.SpeakerEmbedder)
    seen = []
    e.model, e.banks = _ToRecorder(seen), _ToRecorder(seen)
    e.device = torch.device("cpu")
    e._lock = threading.Lock()
    e._want, e._gpu_mem_fraction = want, 0.02
    e.embed = lambda wav: np.ones(256, dtype=np.float32) / 16.0  # warm-up only

    assert e.to_device() == "cuda:0"
    assert calls == [0]
    assert seen == [torch.device("cuda", 0)] * 2 and all(d.index == 0 for d in seen)
    assert e.device == torch.device("cuda", 0)


def test_resolve_device_cpu_paths(monkeypatch):
    torch = pytest.importorskip("torch")
    from vv_longform import embed

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert embed.resolve_device("auto") == torch.device("cpu")
    assert embed.resolve_device("cpu") == torch.device("cpu")


@pytest.mark.skipif(not os.environ.get("VV_TEST_WESPEAKER_PATH"), reason="VV_TEST_WESPEAKER_PATH unset")
def test_real_wespeaker_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from vv_longform import embed

    e = embed.SpeakerEmbedder(os.environ["VV_TEST_WESPEAKER_PATH"], device="cpu")
    assert e.to_device() == "cpu"
    wavs = [w for w in os.environ.get("VV_TEST_SPEECH_WAVS", "").split(",") if w]
    if len(wavs) < 2:
        v = e.embed_pcm24k(voices(3.0, [0]))
        assert v.shape == (256,) and abs(np.linalg.norm(v) - 1) < 1e-4
        return
    clips = [subprocess.run(["ffmpeg", "-loglevel", "error", "-i", w, "-ac", "1", "-ar", "24000", "-f", "s16le", "-"],
                            capture_output=True, check=True).stdout for w in wavs[:2]]  # fmt: skip
    a, b = (np.frombuffer(c, dtype=np.int16) for c in clips)
    h = a.size // 2
    same = float(e.embed_pcm24k(a[:h]) @ e.embed_pcm24k(a[h:]))
    diff = float(e.embed_pcm24k(a[:h]) @ e.embed_pcm24k(b))
    print(f"wespeaker smoke: same-speaker cos={same:.3f} different-speaker cos={diff:.3f}")
    assert same > 0.5 > diff


def test_unembeddable_item_follows_speaker_across_the_cut():
    a, b = unit([1, 0]), unit([0, 1])
    segs0 = [Segment(0, 100, "0", "x"), Segment(100, 299.8, "1", "y")]  # speaker 1 talks into the cut
    segs1 = [Segment(0, 0.1, "0", "z"), Segment(0.5, 200, "1", "w"), Segment(201, 300, "2", "v")]
    items = linking.make_items(0, 0.0, 300.0, segs0, linking.speaker_ranges(segs0, 300.0, 30.0))
    items += linking.make_items(1, 300.0, 300.0, segs1, linking.speaker_ranges(segs1, 300.0, 30.0))
    embs = {(0, "0"): a, (0, "1"): b, (1, "0"): None, (1, "1"): a, (1, "2"): unit([0.7, 0.7])}
    for it in items:
        it.emb = embs[(it.chunk, it.speaker)]
    lab = linking.labels(items, 0.9, 2.0)
    assert lab[(1, "0")] == lab[(0, "1")]  # the 0.1 s sliver continues chunk 0's last speaker
    assert lab[(1, "1")] == lab[(0, "0")]
    assert lab[(1, "2")] not in (lab[(0, "0")], lab[(0, "1")])  # below threshold: new speaker
