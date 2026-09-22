# NVIDIA Nemotron 3 Diarization on Baseten

[NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) answers
"who spoke when" in real-world audio: up to eight speakers, labels ordered by first arrival, and a
single checkpoint that runs at any of several algorithmic latencies, from a 0.32-second input
buffer for live agents to a 30-second buffer for recordings. It is the successor to Streaming
Sortformer v2.1 and is released under the [OpenMDW 1.1](https://openmdw.ai/license/1-1/) license,
which permits commercial use.

Baseten serves the same model three ways. Each preset runs on one NVIDIA RTX PRO 6000 behind a
CUDA-graph engine built around NVIDIA's NeMo streaming loop, so throughput and concurrency exceed
the reference script on the same GPU.

| Preset | Transport | Use it for | Capacity per GPU |
|---|---|---|---|
| [`batch/`](batch/) | HTTP, one request per file | Recorded audio: a file URL or base64 in, speaker turns out | 200 six-minute files per minute, sustained |
| [`streaming/`](streaming/) | WebSocket, one connection per stream | Live audio: 100 ms PCM frames in, a live turn list on every chunk | 560 hour-long streams at the 1.04 s profile, 200 at 0.32 s |
| [`diarized-transcription-streaming/`](diarized-transcription-streaming/) | WebSocket | Live speaker-tagged words: the diarizer paired with NVIDIA's multitalker Parakeet 0.6B ASR (English) | 190 hour-long streams |

A multilingual transcription pairing is in progress and will be added as a fourth preset.

Model Library page: <https://www.baseten.co/library/nemotron-3-diarization>

## Latency profiles

One checkpoint, chosen per request (batch) or per connection (streaming). Input-buffer latency is
(chunk + right context) × 80 ms; compute runs far faster than real time, so the buffer dominates
the time from speech to label.

| Profile | Input-buffer latency | Notes |
|---|---|---|
| `offline` | 30.4 s | Best accuracy. Default for batch. |
| `low` | 1.04 s | Real-time default. |
| `ultralow` | 0.32 s | Lowest latency; three times the step rate of `low`. |

## Accuracy

Diarization error rate with overlapping speech included and no collar, on identical audio and
references for every system. Lower is better. Measured on the General Access checkpoint.

| Dataset | Nemotron 3 `offline` | `low` | `ultralow` | Streaming Sortformer v2.1 (`offline`) |
|---|---|---|---|---|
| NOTSOFAR-1 (129 meetings) | 15.8 | 17.5 | 18.8 | 23.8 |
| AMI-SDM (34 meetings) | 25.0 | 25.5 | 25.9 | 27.8 |
| CALLHOME (12 calls) | 13.0 | 13.8 | 14.0 | 16.7 |
| AISHELL-4, Mandarin (12 meetings) | 10.2 | 9.8 | 11.8 | 27.2 |

Nemotron 3 at its fastest profile beats its predecessor at its slowest on every set. The gain is
largest on recordings with five or more speakers, where v2.1's four-speaker cap costs it.

Real-time diarized transcription (English pairing) scores 28.6 cpWER on NOTSOFAR-1 eval-30 with the
canonical scorer. Words appear a median 0.7 s after they are spoken, speaker labels are final on
arrival, and committed words are never revised.

## Deploying

Each preset directory holds the Truss config Baseten deploys. From the repository root:

```bash
pip install --upgrade truss
truss push nemotron-3-diarization/batch
truss push nemotron-3-diarization/streaming
truss push nemotron-3-diarization/diarized-transcription-streaming
```

The checkpoint is fetched from Hugging Face at deploy time; set the `hf_access_token` secret in
your Baseten workspace if the repository is gated for your account. Every preset needs an
`RTX_PRO_6000` (or `H100`) accelerator.

## Calling the endpoints

Each preset directory documents its full request and response contract and ships runnable
clients (`client.py`, and `curl.sh` for batch). All of them read two environment variables:

```bash
export BASETEN_API_KEY=...   # https://app.baseten.co/settings/account/api_keys
export MODEL_ID=...          # from the model's page in the Baseten dashboard
```

URL forms:

| Preset | Production URL |
|---|---|
| batch | `https://model-{MODEL_ID}.api.baseten.co/environments/production/predict` |
| streaming, diarized transcription | `wss://model-{MODEL_ID}.api.baseten.co/environments/production/websocket` |

Replace `environments/production` with `deployment/{DEPLOYMENT_ID}` to target a specific
deployment. Authenticate with an `Authorization: Api-Key $BASETEN_API_KEY` header.

Audio for the WebSocket presets is 16 kHz mono PCM16, base64-encoded inside JSON text frames
(100 ms frames, 3,200 bytes each, work well). The batch preset accepts any streamable format `ffmpeg`
decodes (WAV, FLAC, MP3, OGG, WebM) and resamples it server-side; see [`batch/README.md`](batch/README.md)
for the MP4/M4A and URL-fetch caveats.

Every client and response in these directories was run against a live deployment of this code on
2026-09-22; the JSON shown is what came back, trimmed only for length.
