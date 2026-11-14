"""Running Nemotron 3.5 Transcribe, and the handle the backend drives it through.

:func:`build` is the entry point: baseten-stt's server imports it by the name in
``BASETEN_STT_BACKEND`` and awaits the backend it returns.

Unlike the Qwen3-ASR template, the engine is not loaded in this process. The model needs NeMo
Speech's vLLM plugin on vLLM 0.32 with a CUDA 13 torch stack, which NVIDIA validates on
Python 3.13 -- while baseten-stt runs on the server image's Python 3.12 with its own pins. So
the engine is the batch preset's ``vllm serve``, launched as a localhost sidecar from its own
venv, and a decode is one chat-completions request to it.
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
import os
import random
import shutil
import signal
import subprocess
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import httpx
from baseten_stt.backend.base import TurnDecodeError

if TYPE_CHECKING:
    from baseten_stt.protocol.capabilities import GrantedContext
    from nemotron_stt.backend import NemotronBackend

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BYTES_PER_SEC = SAMPLE_RATE * 2  # int16 mono
SERVED_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
SIDECAR_PORT = int(os.environ.get("NEMOTRON_SIDECAR_PORT", "8000"))
SIDECAR_URL = f"http://127.0.0.1:{SIDECAR_PORT}"
SIDECAR_SCRIPT = os.environ.get("NEMOTRON_SIDECAR_SCRIPT", "/packages/sidecar/start_vllm.sh")
#: The sidecar loads the mounted weights and initialises the engine: 712 s of engine init on a cold
#: start (no BDN startup cache yet) on vLLM 0.32, less once the cache is restored. Under truss's
#: startup threshold.
SIDECAR_STARTUP_TIMEOUT_S = float(os.environ.get("NEMOTRON_SIDECAR_STARTUP_TIMEOUT_S", "2700"))
#: See ``nemotron_stt.DEFAULT_PARTIAL_WINDOW_S``.
PARTIAL_WINDOW_S = float(os.environ.get("STREAM_PARTIAL_WINDOW_S", "8.0"))
#: The model is a chat model: without an instruction it answers questions about the audio
#: instead of transcribing it. This is the prompt b10-bench sends, which scored 1.5% WER on
#: pipecat and kept non-English speech in its own language.
PROMPT = "Transcribe this audio."
#: The model card's verbatim ASR instruction, with its ``Language: as spoken`` naming one
#: instead. The card forces a language by naming it in the instruction; there is no decoder
#: prefix to pin it, as Qwen3-ASR has.
LANGUAGE_PROMPT = (
    "Audio transcription. Language: {language}. Wording: verbatim, including fillers and "
    "repetitions. Formatting: punctuation and capitalization. Output: transcript only."
)
#: The 40 languages on the model card, by the BCP-47 code the backend declares, to the English
#: name the instruction gives the model.
LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "cs": "Czech", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "lt": "Lithuanian", "lv": "Latvian", "mr": "Marathi", "mt": "Maltese", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "sv": "Swedish", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese", "zh": "Chinese",
}  # fmt: skip
#: The card's languages written in the Latin alphabet. A decode named one of these that comes
#: back in another script did not follow the instruction: on noisy telephone audio named
#: English, Nemotron answered turns in Chinese or Tamil -- background voices, and the caller's
#: own spelled email ("L I 点儿 C O M") -- in 7% of finals (see RETRY_INSTRUCTION).
LATIN_SCRIPT_LANGUAGES = frozenset(
    "cs da de en es et fi fr hr hu id it lt lv mt nl pl pt ro sk sl sv tl tr vi".split()
)
#: Appended to the session's instruction for the one retry of a wrong-script decode.
#: Replaying 69 noisy retail τ-bench calls (voice-agent, client_flush, language en): 175 of 2415
#: finals came back in another script, in 47 calls; the retry turned 28 into English and the rest
#: were dropped. Non-English words 3.7% -> 0.1% of the caller's, words missed 5.9% -> 7.3%, WER
#: 45.8% -> 44.0%, digit recall 89.5% -> 90.1%.
RETRY_INSTRUCTION = (
    "Every word is spoken in {language}: write the transcript in {language}. Do not transcribe "
    "background voices, television, radio, or other conversations."
)
#: Speech runs ~3.3 output tokens per second of audio (11.7k for the 1 hr podcast); the cap is
#: generous against that, and bounds a runaway decode.
TOKENS_PER_AUDIO_S = 12
#: Per decode. A final decodes the whole turn; a long monologue is still seconds, not minutes.
REQUEST_TIMEOUT_S = float(os.environ.get("NEMOTRON_REQUEST_TIMEOUT_S", "120"))
#: The first decodes after the sidecar reports healthy JIT-compile Triton kernels, long enough
#: that a session's finals used to time out. Warmup runs them before the backend is published,
#: so readiness covers them. Turn lengths up to the server's 29 s cut, sequentially, then
#: ``predict_concurrency`` of them at once for the batched shapes. 0 skips warmup.
WARMUP = os.environ.get("NEMOTRON_WARMUP", "1") != "0"
WARMUP_AUDIO_S = (1.0, 4.0, 8.0, 16.0, 29.0)
WARMUP_CONCURRENCY = 32
#: Per warmup decode: the first carries the kernel compiles, not just the audio.
WARMUP_REQUEST_TIMEOUT_S = float(os.environ.get("NEMOTRON_WARMUP_REQUEST_TIMEOUT_S", "900"))
#: How long a server whose sidecar died gets to shut down on its own before it is exited.
SIDECAR_EXIT_GRACE_S = 30.0
#: NEMOTRON_DIAG_WARMUP_PROFILE=1 (diagnostic, default off): run the warmup twice and profile each
#: turn (see _WarmupProfiler); with NEMOTRON_DIAG_RESTART_ONCE=1, start_vllm.sh also runs it against
#: boot 1 (diag_warmup).
DIAG_WARMUP_PROFILE = os.environ.get("NEMOTRON_DIAG_WARMUP_PROFILE", "0") == "1"
#: Seconds into a turn at which the profiler samples; past the last, every DIAG_SAMPLE_EVERY_S.
DIAG_SAMPLE_AT_S = (2.0, 5.0, 10.0)
DIAG_SAMPLE_EVERY_S = 10.0
#: Stack dumps per turn at most (from DIAG_STACKS_FROM_S on), to bound the log volume.
DIAG_MAX_STACK_DUMPS = 6
DIAG_STACKS_FROM_S = 5.0
#: The sidecar-side vLLM histograms whose sums/counts the profiler diffs across a turn.
DIAG_VLLM_METRICS = (
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:request_inference_time_seconds",
)
#: The JIT/compile cache directories (the BDN startup cache's, else each library's default), whose
#: growth across a turn says which compiler ran. Read only; never set here.
DIAG_CACHE_DIRS = (
    ("cuda_driver", "CUDA_CACHE_PATH", "~/.nv/ComputeCache"),
    ("triton", "TRITON_CACHE_DIR", "~/.triton/cache"),
    ("inductor", "TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_root"),
    ("vllm", "VLLM_CACHE_ROOT", "~/.cache/vllm"),
    ("flashinfer", "FLASHINFER_WORKSPACE_BASE", "~/.cache/flashinfer"),
    ("torch_ext", "TORCH_EXTENSIONS_DIR", "~/.cache/torch_extensions"),
    ("numba", "NUMBA_CACHE_DIR", ""),
    ("cute_dsl", "CUTE_DSL_CACHE_DIR", ""),
)


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap int16 mono PCM in a WAV header, which is what the chat route's audio loader reads."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buf.getvalue()


def prompt_for(language: str = "auto", context: GrantedContext | None = None) -> str:
    """The instruction for one decode: in ``language`` (a declared code, or ``"auto"``), with
    the session's context, if it was granted any.

    A session that sets neither gets ``PROMPT`` unchanged, which is what the README's numbers
    were measured with. The context's text follows the instruction as the client wrote it --
    framing for the domain, or a further instruction such as the card's near-field one -- and
    its terms take the card's contextual-ASR form, ``Context terms: ...``.
    """
    if language == "auto":
        prompt = PROMPT
    else:
        prompt = LANGUAGE_PROMPT.format(language=LANGUAGE_NAMES[language])
    if context is not None:
        if context.text:
            prompt += " " + context.text
        if context.keyterms:
            prompt += " Context terms: " + ", ".join(context.keyterms) + "."
    return prompt


def _is_foreign_letter(ch: str) -> bool:
    """Whether ``ch`` is a letter of a script other than Latin.

    Classified by Unicode name rather than code-point ranges, which miss Latin outside the
    Latin blocks (the ``ﬃ`` ligature, fullwidth Latin). Modifier letters (``ʼ`` U+02BC, ``ˈ``)
    are shared marks rather than a script's letters, so they never count; a decode in another
    script still has that script's base letters. Latin-1 letters (``µ``, ``º``) never count.
    """
    if not ch.isalpha() or ord(ch) < 0x100:
        return False
    if unicodedata.category(ch) == "Lm":
        return False
    return "LATIN" not in unicodedata.name(ch, "")


def in_wrong_script(text: str, language: str) -> bool:
    """Whether ``text`` has letters a decode in ``language`` cannot contain. Only Latin-script
    languages are checked; the rest, and ``"auto"``, never are."""
    if language not in LATIN_SCRIPT_LANGUAGES:
        return False
    return any(_is_foreign_letter(ch) for ch in text)


def request_body(pcm: bytes, prompt: str = PROMPT) -> dict:
    """The chat-completions request for one decode of ``pcm`` under ``prompt``."""
    uri = "data:audio/wav;base64," + base64.b64encode(pcm_to_wav(pcm)).decode("ascii")
    return {
        "model": SERVED_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio_url", "audio_url": {"url": uri}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": int(len(pcm) / BYTES_PER_SEC * TOKENS_PER_AUDIO_S) + 32,
        # Already the sidecar's default; stated so the request does not depend on it.
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_response(body: dict) -> str:
    """The transcript from a chat-completions response; empty when the model returned none.

    A decode that ran out of ``max_tokens`` is a truncated transcript -- the budget is ~4x
    the model's rate, so in practice a runaway repetition -- and is refused rather than
    returned as a final: :class:`TurnDecodeError` fails this turn and leaves the session up.
    """
    choices = body.get("choices") or []
    if not choices:
        return ""
    if choices[0].get("finish_reason") == "length":
        raise TurnDecodeError("transcript exceeded the decode budget for this turn's audio")
    return (choices[0].get("message", {}).get("content") or "").strip()


class _SidecarEngine:
    """The model handle the backend holds: one decode, awaited."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def transcribe(
        self, pcm: bytes, language: str = "auto", context: GrantedContext | None = None
    ) -> tuple[str, str | None]:
        """One full decode of ``pcm``. Returns the text and the language it was decoded in:
        the one named, or None under ``"auto"``, since the model reports no detection.

        Cancelling the task awaiting this closes the request's connection, and vLLM aborts a
        request whose client went away -- so a cleared turn frees the GPU.
        """
        prompt = prompt_for(language, context)
        text = await self._decode(pcm, prompt)
        if in_wrong_script(text, language):
            # The instruction names the language but cannot force it, so ask once more, harder.
            # A second miss is not the language the session asked for, so it is no words.
            name = LANGUAGE_NAMES[language]
            retried = await self._decode(pcm, prompt + " " + RETRY_INSTRUCTION.format(language=name))
            logger.info("decode not in %s, retried: %r -> %r", name, text[:80], retried[:80])
            text = "" if in_wrong_script(retried, language) else retried
        return text, None if language == "auto" else language

    async def _decode(self, pcm: bytes, prompt: str) -> str:
        resp = await self._client.post("/v1/chat/completions", json=request_body(pcm, prompt))
        resp.raise_for_status()
        return parse_response(resp.json())


def _wait_for_sidecar(proc: subprocess.Popen, timeout_s: float) -> None:
    """Block until the sidecar answers /health, failing fast if it exits first."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(f"engine sidecar exited with code {code} before becoming healthy")
        try:
            with urllib.request.urlopen(f"{SIDECAR_URL}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    proc.terminate()
    raise RuntimeError(f"engine sidecar not healthy after {timeout_s:.0f}s")


def _warmup_audio(seconds: float) -> bytes:
    """Seeded noise at speech level: silence decodes to nothing, which would leave the decode
    path cold."""
    rng = random.Random(0)
    n = int(seconds * SAMPLE_RATE)
    return b"".join(rng.randint(-3000, 3000).to_bytes(2, "little", signed=True) for _ in range(n))


def _warmup(base_url: str = SIDECAR_URL, profiler: _WarmupProfiler | None = None, label: str = "") -> None:
    """Decode representative turns before the backend is published, failing startup if the
    engine cannot. The transcripts are discarded; only completing matters.

    ``profiler`` (diagnostic) wraps each turn; ``label`` prefixes the timing lines."""
    prefix = f"warmup{label and ' ' + label}"
    clips = [_warmup_audio(s) for s in WARMUP_AUDIO_S]
    started = time.monotonic()
    with httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(WARMUP_REQUEST_TIMEOUT_S, connect=5.0),
        limits=httpx.Limits(max_connections=WARMUP_CONCURRENCY),
    ) as client:

        def decode(pcm: bytes) -> None:
            client.post("/v1/chat/completions", json=request_body(pcm)).raise_for_status()

        def turn(name: str):
            return profiler.turn(f"{prefix} {name}") if profiler else contextlib.nullcontext()

        for seconds, pcm in zip(WARMUP_AUDIO_S, clips):
            t0 = time.monotonic()
            with turn(f"{seconds:.0f}s"):
                decode(pcm)
            logger.info("%s: %.0fs turn in %.1fs", prefix, seconds, time.monotonic() - t0)
        burst = [clips[i % len(clips)] for i in range(WARMUP_CONCURRENCY)]
        t0 = time.monotonic()
        with turn(f"{len(burst)}x"), ThreadPoolExecutor(WARMUP_CONCURRENCY) as pool:
            list(pool.map(decode, burst))
        logger.info("%s: %d concurrent turns in %.1fs", prefix, len(burst), time.monotonic() - t0)
    logger.info("%s done in %.1fs", prefix, time.monotonic() - started)


def _proc_stat(pid: int | str, tid: int | str | None = None) -> tuple[str, int, int] | None:
    """(comm, ppid, utime + stime ticks) of a process (or one of its threads), None if gone."""
    path = f"/proc/{pid}/task/{tid}/stat" if tid is not None else f"/proc/{pid}/stat"
    try:
        with open(path) as f:
            raw = f.read()
    except OSError:
        return None
    comm = raw[raw.index("(") + 1 : raw.rindex(")")]
    fields = raw[raw.rindex(")") + 2 :].split()
    return comm, int(fields[1]), int(fields[11]) + int(fields[12])


def _descendants(root: int) -> list[int]:
    """``root`` and every live process under it."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            st = _proc_stat(entry)
            if st is not None:
                children.setdefault(st[1], []).append(int(entry))
    found, todo = [], [root]
    while todo:
        pid = todo.pop()
        if pid in found or _proc_stat(pid) is None:
            continue
        found.append(pid)
        todo.extend(children.get(pid, []))
    return sorted(found)


def _tree_size(path: str) -> tuple[int, int]:
    """(files, bytes) under ``path``; (0, 0) when it does not exist."""
    files = size = 0
    for dirpath, _, names in os.walk(path):
        for name in names:
            try:
                size += os.lstat(os.path.join(dirpath, name)).st_size
                files += 1
            except OSError:
                pass
    return files, size


def _parse_vllm_metrics(text: str, names=DIAG_VLLM_METRICS) -> dict[str, float]:
    """``<name>_sum`` and ``<name>_count`` of each histogram, summed over label sets."""
    wanted = {f"{n}{suffix}" for n in names for suffix in ("_sum", "_count")}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key = line.split("{", 1)[0].split(" ", 1)[0]
        if key in wanted:
            try:
                out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


class _WarmupProfiler:
    """Diagnostic (NEMOTRON_DIAG_WARMUP_PROFILE=1): what the sidecar does during each warmup turn.

    The decode runs in vLLM's processes, not here, so for each turn this logs, from outside them:

    * vLLM's own request histograms diffed across the turn (``/metrics``): e2e, queue, prefill,
      decode, inference and TTFT seconds. Turn time not in e2e was spent before vLLM's request
      accounting started (input processing in the API server).
    * Samples at DIAG_SAMPLE_AT_S then every DIAG_SAMPLE_EVERY_S while the turn runs: CPU used
      by each sidecar process and its busiest threads since the last sample, and GPU utilisation
      (nvidia-smi, when present). Busy CPU with an idle GPU is a compile or a driver JIT.
    * From DIAG_STACKS_FROM_S on, up to DIAG_MAX_STACK_DUMPS SIGUSR2 stack dumps of every sidecar
      process that registered one (autotune_hook/sitecustomize.py), which name the code running.
    * Growth of the JIT/compile cache directories (DIAG_CACHE_DIRS) across the turn: the CUDA
      driver's (PTX JIT), Triton's, Inductor's, vLLM's, FlashInfer's.

    Request content is never logged: only timings, process names, sizes and stacks.
    """

    def __init__(self, base_url: str, root_pid: int, tag: str) -> None:
        self.base_url = base_url
        self.root_pid = root_pid
        self.tag = tag
        self.marker_dir = os.environ.get("NEMOTRON_DIAG_STACK_DIR", "/tmp/nemotron-diag-stacks")
        self.cache_dirs = []
        for name, var, default in DIAG_CACHE_DIRS:
            path = os.environ.get(var) or (os.path.expanduser(default) if default else "")
            if path:
                self.cache_dirs.append((name, path))
        self.tick_s = 1.0 / os.sysconf("SC_CLK_TCK")
        logger.info("diag[%s] profiler on: sidecar root pid %d, cache dirs %s", tag, root_pid,
                    " ".join(f"{n}={p}" for n, p in self.cache_dirs))

    def _metrics(self) -> dict[str, float]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/metrics", timeout=5) as resp:
                return _parse_vllm_metrics(resp.read().decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001 - diagnostic
            logger.info("diag[%s] metrics unavailable: %s", self.tag, e)
            return {}

    def _caches(self) -> dict[str, tuple[int, int]]:
        return {name: _tree_size(path) for name, path in self.cache_dirs}

    def _cpu(self) -> dict[tuple[int, int | None], tuple[str, int]]:
        """CPU ticks of each sidecar process (tid None) and each of its threads."""
        out = {}
        for pid in _descendants(self.root_pid):
            st = _proc_stat(pid)
            if st is None:
                continue
            out[pid, None] = (st[0], st[2])
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                continue
            for tid in tids:
                tst = _proc_stat(pid, tid)
                if tst is not None:
                    out[pid, int(tid)] = (tst[0], tst[2])
        return out

    def _gpu(self) -> str:
        if not shutil.which("nvidia-smi"):
            return "n/a"
        try:
            return subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().replace("\n", "; ")
        except Exception as e:  # noqa: BLE001 - diagnostic
            return f"n/a ({type(e).__name__})"

    def _dump_stacks(self, name: str, at: float) -> None:
        """SIGUSR2 to each sidecar process that registered a dump and is the process it was."""
        signalled = []
        for pid in _descendants(self.root_pid):
            try:
                with open(os.path.join(self.marker_dir, str(pid))) as f:
                    expected = f.read().strip()
                with open(f"/proc/{pid}/stat") as f:
                    actual = f.read().rsplit(")", 1)[1].split()[19]
            except (OSError, IndexError):
                continue
            if actual == expected:
                signalled.append(pid)
        logger.info("diag[%s] %s +%.0fs: stack dump of pids %s follows", self.tag, name, at, signalled)
        for pid in signalled:
            try:
                os.kill(pid, signal.SIGUSR2)
            except OSError:
                pass

    def _sample(self, name: str, at: float, last: dict) -> dict:
        now = self._cpu()
        span = max(at - last.get("_at", 0.0), 1e-3)
        procs, threads = [], []
        for (pid, tid), (comm, ticks) in now.items():
            if (pid, tid) not in last:
                continue
            pct = 100.0 * (ticks - last[pid, tid][1]) * self.tick_s / span
            (procs if tid is None else threads).append((pct, pid, tid, comm))
        busy = " ".join(f"{comm}[{pid}]={pct:.0f}%" for pct, pid, _, comm in sorted(procs, reverse=True) if pct >= 1)
        top = " ".join(f"{comm}[{tid}]={pct:.0f}%" for pct, _, tid, comm in sorted(threads, reverse=True)[:4] if pct >= 1)
        logger.info("diag[%s] %s +%.0fs: cpu %s | threads %s | gpu %s", self.tag, name, at,
                    busy or "idle", top or "idle", self._gpu())
        now["_at"] = at
        return now

    @contextlib.contextmanager
    def turn(self, name: str):
        metrics0, caches0 = self._metrics(), self._caches()
        cpu0 = self._cpu()
        cpu0["_at"] = 0.0
        done = threading.Event()
        t0 = time.monotonic()

        def sampler() -> None:
            last, dumps, i = cpu0, 0, 0
            while True:
                at = DIAG_SAMPLE_AT_S[i] if i < len(DIAG_SAMPLE_AT_S) else (
                    DIAG_SAMPLE_AT_S[-1] + DIAG_SAMPLE_EVERY_S * (i - len(DIAG_SAMPLE_AT_S) + 1))
                i += 1
                if done.wait(max(0.0, at - (time.monotonic() - t0))):
                    return
                try:
                    last = self._sample(name, at, last)
                    if at >= DIAG_STACKS_FROM_S and dumps < DIAG_MAX_STACK_DUMPS:
                        dumps += 1
                        self._dump_stacks(name, at)
                except Exception as e:  # noqa: BLE001 - diagnostic
                    logger.info("diag[%s] sample failed: %s", self.tag, e)

        thread = threading.Thread(target=sampler, name="diag-warmup-sampler", daemon=True)
        thread.start()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            done.set()
            thread.join(timeout=30)
            try:
                metrics1, caches1 = self._metrics(), self._caches()
                deltas = " ".join(
                    f"{k.removeprefix('vllm:')}=+{metrics1[k] - metrics0.get(k, 0.0):.2f}"
                    for k in sorted(metrics1) if metrics1[k] != metrics0.get(k, 0.0)
                )
                grown = " ".join(
                    f"{n}=+{caches1[n][0] - caches0[n][0]}f/+{(caches1[n][1] - caches0[n][1]) / 2**20:.1f}MiB"
                    f"(now {caches1[n][1] / 2**20:.1f}MiB)"
                    for n in caches1 if caches1[n] != caches0.get(n)
                )
                logger.info("diag[%s] %s: %.2fs | vllm %s | caches grown %s", self.tag, name, elapsed,
                            deltas or "unchanged", grown or "none")
            except Exception as e:  # noqa: BLE001 - diagnostic
                logger.info("diag[%s] %s: %.2fs (summary failed: %s)", self.tag, name, elapsed, e)


def _profiled_warmups(base_url: str, root_pid: int, tag: str) -> None:
    """The diagnostic warmup: twice in this process, every turn profiled. Pass 1 is the cold one
    (fails startup as the normal warmup would); pass 2 shows what this process's caches fix."""
    profiler = _WarmupProfiler(base_url, root_pid, tag)
    times = []
    for n in (1, 2):
        t0 = time.monotonic()
        _warmup(base_url, profiler, label=f"{tag} pass{n}")
        times.append(time.monotonic() - t0)
    logger.info("diag[%s] warmup pass1 %.1fs, pass2 %.1fs", tag, *times)


def diag_warmup(port: int, root_pid: int, tag: str) -> None:
    """The profiled warmup against another vLLM (start_vllm.sh's boot 1 under
    NEMOTRON_DIAG_RESTART_ONCE=1 and NEMOTRON_DIAG_WARMUP_PROFILE=1), from its own interpreter."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    try:
        _profiled_warmups(f"http://127.0.0.1:{port}", root_pid, tag)
    except Exception:
        logger.exception("diag[%s] warmup failed", tag)
        raise


def _supervise(proc: subprocess.Popen) -> None:
    """Take the server down with the sidecar.

    baseten-stt's probes only say the backend was built, so a replica whose vLLM died would
    stay ready and fail every decode. SIGTERM lets uvicorn shut down; the hard exit covers a
    shutdown that stalls on open sessions. Either way the process ends and the platform
    replaces the replica.
    """
    code = proc.wait()
    logger.error("engine sidecar exited with code %s; stopping the server", code)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(SIDECAR_EXIT_GRACE_S)
    logger.error("server still up %.0fs after the sidecar exited; exiting", SIDECAR_EXIT_GRACE_S)
    os._exit(1)


def build() -> NemotronBackend:
    """The factory ``BASETEN_STT_BACKEND=nemotron_stt:build`` resolves to.

    Synchronous on purpose, like the template's: bringing the engine up blocks for minutes,
    and a coroutine would only move that block onto the event loop about to serve sessions.
    The sidecar inherits this process's stdout and stderr, so its download and vLLM logs land
    in the deployment logs beside the server's.
    """
    from nemotron_stt.backend import NemotronBackend

    logger.info("starting engine sidecar: %s", SIDECAR_SCRIPT)
    proc = subprocess.Popen(["/bin/bash", SIDECAR_SCRIPT])
    _wait_for_sidecar(proc, SIDECAR_STARTUP_TIMEOUT_S)
    if os.environ.get("NEMOTRON_DIAG_RESTART_ONCE") == "1":
        # start_vllm.sh's diagnostic second start answered /health (boot 1 used a private port).
        logger.info("diag restart: boot2 ready")
    if WARMUP:
        try:
            if DIAG_WARMUP_PROFILE:
                tag = "boot2" if os.environ.get("NEMOTRON_DIAG_RESTART_ONCE") == "1" else "boot"
                _profiled_warmups(SIDECAR_URL, proc.pid, tag)
            else:
                _warmup()
        except Exception:
            proc.terminate()
            raise
    threading.Thread(target=_supervise, args=(proc,), name="sidecar-supervisor", daemon=True).start()
    client = httpx.AsyncClient(
        base_url=SIDECAR_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=5.0),
        # One connection per concurrent decode, finals and partials alike; vLLM batches them.
        limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
    )
    logger.info("Nemotron 3.5 Transcribe sidecar ready, serving the baseten-stt protocol")
    return NemotronBackend(_SidecarEngine(client), partial_window_s=PARTIAL_WINDOW_S)
