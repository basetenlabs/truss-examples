"""Batch selection for BATCH_LENGTH_BUCKETING in model/model.py.

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with prometheus-client pytest stt/parakeet-tdt-0.6b-v3/batch/tests

CPU only: these tests form batches from queued jobs and never call NeMo or a GPU.
They cover which jobs a batch takes, that the oldest job always runs, and that jobs
not chosen stay pending in arrival order and are never dropped.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from pathlib import Path

import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.py"

# The deployed config's values for everything the module reads at import time.
os.environ.update(
    {
        "MAX_BATCH_SIZE": "4",
        "BATCH_WINDOW_MS": "0",
        "BATCH_BUCKET_SECONDS": "2,4,8",
    }
)
_spec = importlib.util.spec_from_file_location("parakeet_tdt_model_bucketing", MODEL_PATH)
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def _cap(limit):
    return lambda _seconds: limit


def test_oldest_always_chosen_and_closest_fill():
    lengths = [5.0, 1.0, 5.2, 9.0, 4.9, 5.1, 20.0]
    # 5.1 and 4.9 tie at 0.1 from the oldest; arrival order breaks the tie.
    assert mm._closest_length_batch(lengths, _cap(4)) == [0, 2, 4, 5]


def test_ties_break_in_arrival_order():
    assert mm._closest_length_batch([3.0, 2.0, 4.0, 2.0, 4.0], _cap(3)) == [0, 1, 2]


def test_oldest_chosen_even_when_it_is_the_outlier():
    assert mm._closest_length_batch([30.0, 1.0, 1.0, 1.0], _cap(2)) == [0, 1]


def test_single_and_cap_one():
    assert mm._closest_length_batch([7.0], _cap(16)) == [0]
    assert mm._closest_length_batch([7.0, 7.0, 7.0], _cap(1)) == [0]


def test_memory_cap_follows_longest_clip():
    # Cap is 3 while the longest clip is under 100 s, 1 at or above it.
    def cap(longest):
        return 3 if longest < 100 else 1

    # 150 s is skipped (it would shrink the cap below the batch size); later,
    # farther clips that keep the cap still join.
    assert mm._closest_length_batch([90.0, 150.0, 20.0, 10.0], cap) == [0, 2, 3]


def _efficiency(lengths, chosen):
    picked = [lengths[i] for i in chosen]
    return sum(picked) / (len(picked) * max(picked))


def test_padding_cap_skips_clips_that_would_pad_too_much():
    lengths = [5.0, 5.2, 4.0, 9.0, 4.9, 2.0]
    # Without the cap the four closest clips join, padding efficiency ~0.92.
    assert mm._closest_length_batch(lengths, _cap(4)) == [0, 1, 2, 4]
    # At 0.95, 4.0 s would drop the batch to ~0.92, so it is skipped; farther
    # clips are still tried and 9.0 / 2.0 fail too.
    chosen = mm._closest_length_batch(lengths, _cap(4), 0.95)
    assert chosen == [0, 1, 4]
    assert _efficiency(lengths, chosen) >= 0.95


@pytest.mark.parametrize("threshold", [0.5, 0.8, 0.9, 0.99, 1.0])
def test_padding_cap_is_respected(threshold):
    lengths = [1.0 + 0.37 * ((7 * i) % 23) for i in range(40)]
    chosen = mm._closest_length_batch(lengths, _cap(16), threshold)
    assert chosen[0] == 0
    assert len(chosen) == 1 or _efficiency(lengths, chosen) >= threshold - 1e-9


def test_padding_cap_always_takes_the_oldest():
    # Every other clip is far from the 30 s oldest clip; it still runs, alone.
    assert mm._closest_length_batch([30.0, 1.0, 1.0, 1.0], _cap(4), 0.9) == [0]
    # A cap of 1.0 admits only clips of identical length.
    assert mm._closest_length_batch([3.0, 3.0, 3.1, 3.0], _cap(4), 1.0) == [0, 1, 3]


def test_zero_padding_cap_changes_nothing():
    lengths = [5.0, 1.0, 5.2, 9.0, 4.9, 5.1, 20.0]
    assert mm._closest_length_batch(lengths, _cap(4), 0.0) == mm._closest_length_batch(
        lengths, _cap(4)
    )


def test_real_cap_is_max_batch_size():
    # This preset has no per-length memory cap.
    assert mm.Model._length_batch_cap(0.0) == mm.MAX_BATCH_SIZE
    assert mm.Model._length_batch_cap(1440.0) == mm.MAX_BATCH_SIZE


def _job(seconds, *, timestamps=False):
    return {
        "waveform": None,
        "bucket": mm.Model._duration_bucket(seconds),
        "duration_seconds": seconds,
        "timestamps": timestamps,
        "done": threading.Event(),
        "enqueued_at": time.monotonic(),
    }


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(mm, "BATCH_LENGTH_BUCKETING", True)
    monkeypatch.setattr(mm, "BATCH_BUCKET_LOOKAHEAD", 8)
    return mm.Model()


def _form(model):
    return model._length_bucketed_batch(model._next_job())


def test_batch_takes_closest_and_keeps_the_rest_in_order(model):
    # 2.0 and 1.9 s fall in the coarse 2 s bucket, 2.1 and 2.3 s in the 4 s one;
    # the default batcher would split them, length bucketing batches them.
    jobs = [_job(s) for s in (2.0, 9.0, 2.1, 1.0, 1.9, 2.3, 7.0)]
    for job in jobs:
        model._jobs.put(job)
    batch = _form(model)
    assert [job["duration_seconds"] for job in batch] == [2.0, 2.1, 1.9, 2.3]
    assert [job["duration_seconds"] for job in model._deferred_jobs] == [9.0, 1.0, 7.0]
    assert all("selected_at" in job for job in batch)
    assert not any("selected_at" in job for job in model._deferred_jobs)


def test_lookahead_bounds_the_pool(model, monkeypatch):
    monkeypatch.setattr(mm, "BATCH_BUCKET_LOOKAHEAD", 3)
    for seconds in (5.0, 1.0, 1.0, 5.0, 5.0):
        model._jobs.put(_job(seconds))
    batch = _form(model)
    # Only the first three jobs are considered; the two 5 s clips stay queued.
    assert [job["duration_seconds"] for job in batch] == [5.0, 1.0, 1.0]
    assert model._jobs.qsize() == 2


def test_timestamp_jobs_batch_separately(model):
    for job in (_job(3.0), _job(3.0, timestamps=True), _job(30.0), _job(3.0)):
        model._jobs.put(job)
    first = _form(model)
    assert [(j["duration_seconds"], j["timestamps"]) for j in first] == [
        (3.0, False),
        (30.0, False),
        (3.0, False),
    ]
    second = _form(model)
    assert [(j["duration_seconds"], j["timestamps"]) for j in second] == [(3.0, True)]
    assert not model._deferred_jobs and model._jobs.empty()


@pytest.mark.parametrize("padding_cap", [0.0, 0.85])
def test_every_job_runs_once_and_waits_boundedly(model, monkeypatch, padding_cap):
    monkeypatch.setattr(mm, "BATCH_MIN_PADDING_EFFICIENCY", padding_cap)
    # Three arrivals per batch against a batch size of 4, mostly ~1 s clips with an
    # occasional 12 s outlier that is never close to the oldest short clip. Each
    # outlier must still run once it is the oldest pending job, within the lookahead.
    lookahead = mm.BATCH_BUCKET_LOOKAHEAD
    arrived_at, waits = {}, []
    submitted = 0
    for step in range(300):
        for k in range(3):
            seconds = 12.0 if (step + k) % 17 == 0 else 1.0 + 0.1 * ((step + k) % 4)
            job = _job(seconds)
            arrived_at[id(job)] = step
            submitted += 1
            model._jobs.put(job)
        batch = _form(model)
        assert 1 <= len(batch) <= mm.MAX_BATCH_SIZE
        if padding_cap:
            seconds_in_batch = [job["duration_seconds"] for job in batch]
            assert len(batch) == 1 or sum(seconds_in_batch) >= padding_cap * len(
                batch
            ) * max(seconds_in_batch)
        assert len(model._deferred_jobs) < lookahead
        waits.extend(step - arrived_at.pop(id(job)) for job in batch)
    drained = 0
    while not (model._jobs.empty() and not model._deferred_jobs):
        for job in _form(model):
            arrived_at.pop(id(job))
            drained += 1
    assert not arrived_at
    assert len(waits) + drained == submitted
    assert max(waits) <= lookahead


def test_default_mode_is_unchanged(monkeypatch):
    monkeypatch.setattr(mm, "BATCH_LENGTH_BUCKETING", False)
    # The default batcher only takes jobs that arrive within the window.
    monkeypatch.setattr(mm, "BATCH_WINDOW_SECONDS", 0.05)
    model = mm.Model()
    for seconds in (3.0, 9.0, 3.1, 1.0, 2.9, 3.3):
        model._jobs.put(_job(seconds))
    batch = model._arrival_order_batch(model._next_job())
    # Arrival order within the first clip's 4 s bucket; 1.0 s and 9.0 s deferred.
    assert [job["duration_seconds"] for job in batch] == [3.0, 3.1, 2.9, 3.3]
    assert [job["duration_seconds"] for job in model._deferred_jobs] == [9.0, 1.0]


# BATCH_WINDOW_EARLY_EXIT: stop waiting out the batch window once nothing in the pod
# can join the batch. Waits are bounded by a long window so the on/off contrast is
# far larger than scheduler jitter.
LONG_WINDOW_S = 0.4


def _timed_form(model, form):
    started = time.monotonic()
    batch = form(model._next_job())
    return batch, time.monotonic() - started


@pytest.fixture
def windowed(monkeypatch):
    monkeypatch.setattr(mm, "BATCH_WINDOW_SECONDS", LONG_WINDOW_S)
    monkeypatch.setattr(mm, "BATCH_LENGTH_BUCKETING", True)
    monkeypatch.setattr(mm, "BATCH_BUCKET_LOOKAHEAD", 8)
    return mm.Model()


@pytest.mark.parametrize("batcher", ["_length_bucketed_batch", "_arrival_order_batch"])
def test_default_waits_out_the_window(windowed, batcher):
    assert mm.BATCH_WINDOW_EARLY_EXIT is False
    windowed._jobs.put(_job(3.0))
    batch, elapsed = _timed_form(windowed, getattr(windowed, batcher))
    assert len(batch) == 1
    assert elapsed >= LONG_WINDOW_S * 0.9


@pytest.mark.parametrize("batcher", ["_length_bucketed_batch", "_arrival_order_batch"])
def test_early_exit_skips_the_window_when_nothing_is_incoming(
    windowed, monkeypatch, batcher
):
    monkeypatch.setattr(mm, "BATCH_WINDOW_EARLY_EXIT", True)
    for seconds in (3.0, 3.1, 2.9):
        windowed._jobs.put(_job(seconds))
    batch, elapsed = _timed_form(windowed, getattr(windowed, batcher))
    # Everything already queued still joins; only the idle wait is skipped.
    assert [job["duration_seconds"] for job in batch] == [3.0, 3.1, 2.9]
    assert elapsed < LONG_WINDOW_S / 4


@pytest.mark.parametrize("batcher", ["_length_bucketed_batch", "_arrival_order_batch"])
def test_early_exit_waits_for_a_clip_still_being_admitted(windowed, monkeypatch, batcher):
    monkeypatch.setattr(mm, "BATCH_WINDOW_EARLY_EXIT", True)
    windowed._jobs.put(_job(3.0))
    late = _job(3.2)
    with windowed._admitting_lock:
        windowed._admitting += 1

    def admit():
        time.sleep(0.05)
        windowed._jobs.put(late)
        windowed._leave_admission()

    thread = threading.Thread(target=admit)
    thread.start()
    batch, elapsed = _timed_form(windowed, getattr(windowed, batcher))
    thread.join()
    assert [job["duration_seconds"] for job in batch] == [3.0, 3.2]
    # Joined as soon as it was enqueued, then nothing else was incoming.
    assert elapsed < LONG_WINDOW_S / 2
    assert windowed._admitting == 0


def test_admission_count_returns_to_zero(monkeypatch):
    import numpy as np

    model = mm.Model()
    monkeypatch.setattr(mm, "PREDICT_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(model, "_decode_audio_input", lambda request: b"audio")
    monkeypatch.setattr(
        mm.Model, "_decode_waveform", staticmethod(lambda _: np.zeros(16000, np.float32))
    )
    seen = {}

    def worker():
        job = model._jobs.get()
        time.sleep(0.02)  # predict() leaves admission right after put()
        seen["admitting_while_queued"] = model._admitting
        job["output"] = type("Hypothesis", (), {"text": "ok", "timestamp": None})()
        job["inference_finished_at"] = time.monotonic()
        job["done"].set()

    thread = threading.Thread(target=worker)
    thread.start()
    assert model._predict_impl({"audio_b64": ""})["text"] == "ok"
    thread.join()
    # A clip waiting for the GPU is no longer "incoming".
    assert seen["admitting_while_queued"] == 0
    assert model._admitting == 0

    def fail(request):
        raise ValueError("bad audio")

    monkeypatch.setattr(model, "_decode_audio_input", fail)
    with pytest.raises(ValueError):
        model._predict_impl({"audio_b64": ""})
    assert model._admitting == 0
    with pytest.raises(RuntimeError, match="negative"):
        model._leave_admission()


def test_zero_frame_wav_is_rejected_on_the_soundfile_path(monkeypatch):
    import io

    import numpy as np
    import soundfile as sf

    def no_ffmpeg(_):
        raise AssertionError("a 16 kHz WAV must decode in-process")

    monkeypatch.setattr(mm.Model, "_decode_with_ffmpeg", staticmethod(no_ffmpeg))

    def wav(samples):
        buf = io.BytesIO()
        sf.write(buf, samples, mm.AUDIO_SAMPLE_RATE_HZ, format="WAV")
        return buf.getvalue()

    # Rejected before batching, so it fails this request alone.
    with pytest.raises(ValueError, match="decoded audio has no samples"):
        mm.Model._decode_waveform(wav(np.zeros(0, np.float32)))
    assert mm.Model._decode_waveform(wav(np.zeros(160, np.float32))).shape == (160,)
