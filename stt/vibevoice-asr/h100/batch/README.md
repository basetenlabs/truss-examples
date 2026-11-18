# VibeVoice-ASR

[Microsoft VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) is a multimodal speech-to-text model deployed via [vLLM](https://github.com/vllm-project/vllm) with an OpenAI-compatible chat completions API. It transcribes audio into JSON-shaped segments with speaker labels and timestamps.

## Hardware

- **GPU:** H100 (single)
- **System memory:** 32 GiB
- **Cold start:** ~3 minutes (HF snapshot + tokenizer file generation + vLLM warmup)

## Deployment

```bash
baseten model push
```

The model weights are public and are downloaded without a Hugging Face access token.

## API

The deployment exposes the OpenAI chat completions endpoint. Point any OpenAI-compatible client at it:

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="vibevoice",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."},
        {"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"}},
            {"type": "text", "text": "Transcribe this audio."},
        ]},
    ],
    max_tokens=256,
    temperature=0,
)

print(response.choices[0].message.content)
```

The model returns a JSON-stringified array of segments in the assistant message content:

```json
[
  {"Start": 0.0, "End": 10.46, "Speaker": 0,
   "Content": "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."}
]
```

You can also pass audio inline as a base64 data URI: `data:audio/wav;base64,<...>`.

With the default settings the proxy replaces the text prompt (see below), so any prompt works, and lowers
`max_tokens` to the loop guard when yours is higher. Hotwords survive the rewrite: put them in a text part as
`with extra info: <hotwords>` (the upstream form) and they go into the prompt. Set `VV_CANONICAL_PROMPT=0` to send
your own prompt and `VV_MAX_TOKENS_PER_S=0` to turn the guard off.

## Long audio: how a request is served

A small proxy (`packages/vv_longform`) listens on port 8000 and runs vLLM on 127.0.0.1:8001. It passes every
vLLM route through and adds three things in front of the model:

1. **The model's own prompt** (`VV_CANONICAL_PROMPT=1`). Every audio request on both routes is rewritten to
   the upstream recipe (microsoft/VibeVoice `vllm_plugin/tests/test_api.py`): the upstream system prompt, then
   the audio, then "This is a N seconds audio, please transcribe it with these keys: Start time, End time,
   Speaker ID, Content" (with "with extra info: <hotwords>" inserted when the client sent hotwords).
2. **A loop guard on output length** (`VV_MAX_TOKENS_PER_S=12`, `VV_MAX_TOKENS_BASE=256`): a single-pass request
   gets `max_tokens = min(12 x audio seconds + 256, its own max_tokens)`, and each chunk of a chunked request
   gets `12 x chunk seconds + 256`; both are clamped to the context window. Normal transcripts use about 5
   tokens per audio second (7-10 on dense meetings), so this only cuts a rare repetition loop short.
3. **Parallel chunks for long files** (`VV_CHUNK_S=300`). Audio longer than ~330 s is cut at the quietest 0.5 s
   frame within 20 s of every 300 s mark (no overlap), the chunks decode as parallel requests in one vLLM
   batch, and their segments are stitched back with absolute timestamps. A 1-hour file becomes 13 chunks.
   Single-pass decoding of a full hour is slow and can loop; chunking keeps every chunk short.

**Stop tokens** live in the checkpoint directory, not the proxy. The checkpoint ships no
`generation_config.json`, so vLLM stopped only on `<|endoftext|>` (151643) while the model ends its turn with
`<|im_end|>` (151645), and every reply ran to `max_tokens`. `data/start.sh` writes `generation_config.json`
with `eos_token_id: [151643, 151645]` next to the tokenizer files (merging into a fine-tune's own file), and
vLLM adds both ids to every request's stop tokens, on every route, with or without the proxy. The model emits
`<|im_end|>` only after the recipe prompt, so `VV_CANONICAL_PROMPT=0` and `VV_PROXY=0` also need the client to
send that prompt.

**Speaker linking across chunks** (`VV_LINK=embed`). Each chunk labels its own speakers. As soon as a chunk
returns, the proxy embeds up to 30 s of each of that chunk's speakers with WeSpeaker ResNet34-LM (45 MB,
mounted at `/models/wespeaker-resnet34-lm`, run on the GPU), then clusters all chunk speakers by cosine
similarity (average linkage, never merging two speakers of the same chunk, stopping below
`VV_LINK_THRESHOLD`, 0.30). Labels are renumbered by first appearance, like a single-pass reply. A speaker with
under 2 s of clean speech in a chunk joins its most similar cluster. Linking adds about 0.1 s per request.
The linker loads at startup; `/v1/models` returns 503 until it is ready.

**Audio preparation** (`VV_PREP_PARALLEL=1`). The proxy fetches and decodes audio itself (any ffmpeg format,
including m4a URLs). For chunked requests it decodes the file as 4 parallel ffmpeg segments and checks each
seam against the next segment; any mismatch falls back to a sequential decode. AAC coded with PNS can differ
by a few hundred noise samples per minute at a seam; other audio decodes sample-identical.

## Speaker-attributed transcription (`/v1/audio/transcriptions`)

```bash
curl -s https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1/audio/transcriptions \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -F model=vibevoice -F response_format=verbose_json -F file=@meeting.flac
# {"duration": 61.2, "text": "...", "segments": [{"start": 0.0, "end": 4.1, "speaker": "S0", "text": "..."}, ...]}
```

Each audio request logs one `timing` line: `audio_s`, `n_chunks`, prompt/completion tokens, `finish_reason`,
`prep_ms`/`queue_ms`/`gen_ms`/`link_ms`/`total_ms`, and for chunked requests `link`, `n_speakers`, `embed_ms`,
`fetch_ms`, `decode_ms`, `prep_mode` (`seq`/`parallel`/`fallback`/`loose`), `n_seg`, `seam_mismatch`.

## Performance (Model Scout long-audio study, PR #1337)

Matched pair on one H100 `/batch-bench` request (same cluster), baseline = the previous single-pass setup with the recipe
prompt and 64k context (eager, no chunking), candidate = these defaults. Server-side p50; xRT = audio seconds
per wall second.

| Workload | Candidate vs baseline |
|---|---|
| 1-hour podcast, c1 | completes in 17 s (xRT 200); the single pass never finishes (runs to the token cap, ~5 min) |
| AMI meetings (14-50 min), c2 | tcpWER 22.0% -> 18.9% (median 18.81%, range 18.32-19.16% over 5 runs); the 16-meeting set runs 3.5x faster |
| 5-min podcast, c1 / c16 | 19% / 5% lower latency, xRT 24 -> 29 / 298 -> 314 |
| Short clips (pipecat ~10 s), c1 / c16 | 17% lower latency, throughput +20% / +15% |

Before the study, the shipped preset (32k context, client prompt, eager) never stopped generating: every
reply hit `max_tokens`, so short clips took ~4 s and AMI tcpWER was 63.5%.

## Tuning knobs (environment variables)

`data/start.sh`: `VV_MAX_MODEL_LEN` (65536), `VV_NUM_GPU_BLOCKS` (12288 = 196k tokens of KV, ~10.5 GiB; 0 =
vLLM sizes the pool), `VV_ENFORCE_EAGER` (0 = CUDA graphs), `VV_MAX_NUM_SEQS` (16),
`VV_GPU_MEMORY_UTILIZATION` (0.85) and, unset by default, `VV_MAX_NUM_BATCHED_TOKENS`,
`VV_MM_PROCESSOR_CACHE_GB`, `VV_EXTRA_VLLM_ARGS`. `VV_PROXY=0` serves vLLM directly on port 8000 (no
transcriptions route, no long-audio handling). vLLM still stops on `<|im_end|>`, but the model only ends its
turn when it gets the upstream recipe prompt: with another prompt it repeats the transcript until `max_tokens`
(exec 15: 200 of 200 short clips hit `finish_reason=length` with "Transcribe this audio."). With `VV_PROXY=0`,
send the recipe yourself.

Proxy (`packages/vv_longform`): `VV_CANONICAL_PROMPT` (1), `VV_MAX_TOKENS_PER_S` (12; 0 = keep the
client's `max_tokens`), `VV_MAX_TOKENS_BASE` (256), `VV_MAX_TOKENS_CAP` (65536), `VV_CHUNK_S` (300; 0 = one pass),
`VV_CHUNK_TRIGGER_S` (1.1 x chunk), `VV_CUT_SEARCH_S` (20), `VV_MAX_INFLIGHT_CHUNKS` (32), `VV_LINK`
(embed | none: chunk-local labels `c<i>s<k>`), `VV_LINK_THRESHOLD` (0.30), `VV_LINK_MAX_S` (30),
`VV_LINK_MIN_S` (2), `VV_LINK_DEVICE` (auto), `VV_PREP_PARALLEL` (1), `VV_PREP_THREADS` (4),
`VV_PREP_MIN_SEG_S` (120), `VV_PREP_SEAM_FALLBACK` (1).

Measured but not adopted: 150 s chunks with 32 sequences (1.5x faster decode on the 1-hour file but AMI tcpWER
+1 pt from more chunk boundaries); speaker-link threshold 0.40 (more over-split speakers).

## Train-to-serve

Serve a fine-tune by swapping `weights.source` (add `auth_secret_name: hf_access_token` for a private repo)
or by mounting a training checkpoint and setting `BASETEN_MODEL_PATH`. A read-only checkpoint gets a
writable directory of symlinks for the plugin's tokenizer files. The proxy, prompt and chunking settings
apply unchanged; a fine-tune trained on a different prompt should set `VV_CANONICAL_PROMPT=0`.

## Known limitations

- **m4a in a base64 data URI is decoded by the proxy** (vLLM 0.14.1's plugin loader can't decode m4a from a buffer). With `VV_PROXY=0`, send WAV or FLAC.
- **One audio clip per request.** The model is trained on single-audio chat messages; multiple `audio_url` items in one user turn are not supported.
- **Speaker counts across chunks can over-split.** On AMI, linking sometimes reports 5-6 speakers for a 4-speaker meeting (never fewer). Speaker confusion drops (3.7% -> 2.4%), but the speaker-count error rises (MAE 0.25 -> 0.375).
- **Streaming requests are never chunked.** A chat request with `stream: true` goes to vLLM as one pass, which
  for long audio is slow and can loop to the token cap (a 1-hour file took ~5 minutes in the study). Send audio
  longer than ~330 s without `stream`.
- **Very long single requests are slow to fetch.** The proxy fetches remote audio itself; an occasional slow fetch adds directly to latency.

## About the patch.py

`data/patch.py` applies three small in-place fixes to the installed `vllm_plugin/model.py` at container boot:

1. **KV-cache delegation forwarders** on `VibeVoiceForCausalLM` — required for vLLM 0.14.1. Without these, vLLM sees zero attention layers and crashes at startup with `IndexError: list index out of range` on `available_gpu_memory[0]`.
2. **`get_data_parser` on the Info class** — forward-compatibility shim for vLLM 0.21+ which calls the parser on the Info class instead of the Processor class.
3. **`mm_data_items` rename try/except** — forward-compatibility shim for vLLM 0.15+ which renamed the `ProcessorInputs.mm_data` field.

These are upstream plugin issues. When Microsoft fixes them, delete `data/patch.py` and remove the `python3 /app/data/patch.py` line from `config.yaml`'s `start_command`.

The patch script is idempotent and uses `assert`-checked anchor strings, so a plugin upstream change that breaks the anchors causes a clear failure at startup rather than silent miscompile.