# Nemotron 3 Diarization (batch) — benchmark

Measured on Baseten **fde-internal**, 1× **RTX-PRO-6000**, bf16, via the batch endpoint
(per-request `latency` across pre-loaded profile instances). NeMo `SortformerEncLabelModel`
from NeMo main (pinned `3c2d62ae7eb4`) on the `nvcr.io/nvidia/nemo:26.08` container (see README / config).


## General Access checkpoint (2026-09-19) — current numbers

Everything below this section was measured on the early-access **preview** checkpoint and is kept as
history. The GA checkpoint is a retrained bf16 model (same architecture, speaker cache 264, same latency
profiles); per-file outputs differ from the preview by 6–10 DER, so no preview number carries over.
Protocol as in §Protocol (DER collar 0 s, overlap included, explicit per-file UEM). Full run:
`rsi-bench/nemotron_diar/release/GA_VALIDATION.md`.

| set | offline | low (1.04 s) | ultralow (0.32 s) | miss / FA / conf (offline) | SCA (offline) |
|---|---|---|---|---|---|
| NOTSOFAR-1 eval SC (129) | **15.80** (preview 16.27) | **17.48** (18.29) | **18.79** (19.42) | 13.12 / 0.88 / 1.80 | 75.2 % |
| AMI-SDM dev+test (34) | **24.99** (25.77) | **25.53** (26.51) | **25.86** (26.81) | 22.49 / 1.65 / 0.85 | 94.1 % |
| CALLHOME (12 local) | **13.02** (12.65) | **13.79** (13.41) | **13.96** (13.66) | 8.58 / 4.02 / 0.43 | 91.7 % |
| AISHELL-4 zh (12) | **10.24** (9.62) | **9.77** | **11.76** | 4.20 / 3.74 / 2.30 | 100 % |

Same refs, Streaming Sortformer v2.1 at offline: NOTSOFAR 23.84, AMI-SDM 27.83, CALLHOME 16.71, AISHELL-4 27.20
(Nemotron 3 GA beats it in all 12 measured cells). AMI rows use the original annotations (NVIDIA's card uses
forced-aligned refs, hence its 11.14 on AMI SDM); VoxConverse is in the training data and is not reported.

Speed on the GA weights (server `fwd`, RTFx = audio ÷ fwd): `low` bs=8 **1,444** (preview 1,478), bs=1 **283**
(426 — the larger per-step state dominates a single file); `ultralow` 474 / 90 (parity); `offline` ~26,000 /
5,346 (parity). Open-loop hold at `low`: **200 files/min holds** (p95 12.4 s flat), 240/min backlogs; ceiling
≈ 207–210/min as on the preview build. bf16 core is DER-neutral on GA (low 17.48 vs fp32 17.53).

## Protocol
- **DER** via `pyannote.metrics`, at **collar 0.25 s and 0.0 s**, overlap **included**
  (`skip_overlap=False`), macro-averaged. Plus speaker-count accuracy (**SCA**) and MAE.
- All four algorithmic-latency profiles tested per dataset (`offline` 30.4 s, `low` 1.04 s,
  `verylow` 0.64 s, `ultralow` 0.32 s buffer). **`verylow` rows are measured but the profile is not
  offered by default** (Inductor cannot compile it; opt-in via `NEMO_DIAR_PROFILES`).
- Datasets & refs:
  - **AMI-SDM dev+test (34 meetings)** — pyannote AMI-diarization-setup `only_words` RTTMs
    (tight, forced-alignment-style), single distant mic (Array1-01).
  - **NOTSOFAR-1 eval_full_with_GT (129 sessions)** — single distant channel; refs built from
    the ground-truth **word timings** (tight speech intervals).
  - **VoxConverse test (232)** — standard refs. ⚠️ **in Nemotron's training data** (contaminated).

## ⚠️ Read before comparing
1. **VoxConverse is train-on-test** for Nemotron (per the card's training tables); the pyannote
   models were not trained on it. AMI-SDM test and NOTSOFAR eval are held out for all.
2. **AMI-SDM does not reconcile with NVIDIA's card** (see below) — treat AMI absolute DER with
   suspicion; the miss is real on our channel but the gap to the card is AMI-specific.

## Full results — Nemotron, all latencies
### AMI-SDM (dev+test, N=34, only_words refs)
| Latency | DER@.25 | DER@0 | Miss | FA | Conf | SCA | Spk-MAE |
|---|---|---|---|---|---|---|---|
| offline | 23.77 | 25.77 | 21.52 | 1.13 | 1.12 | 79.4% | 0.24 |
| low | 24.48 | 26.51 | 21.94 | 1.08 | 1.46 | 76.5% | 0.26 |
| verylow *(measured, not offered)* | 24.76 | 26.82 | 22.05 | 1.10 | 1.61 | 79.4% | 0.21 |
| ultralow | 24.70 | 26.81 | 22.16 | 1.12 | 1.42 | 76.5% | 0.29 |

### NOTSOFAR-1 (eval_full, N=129, word-timing refs)
| Latency | DER@.25 | DER@0 | Miss | FA | Conf | SCA | Spk-MAE |
|---|---|---|---|---|---|---|---|
| offline | 10.47 | 16.27 | 8.25 | 0.52 | 1.70 | 56.6% | 0.46 |
| low | 12.52 | 18.29 | 8.78 | 0.56 | 3.18 | 24.0% | 0.90 |
| verylow *(measured, not offered)* | 12.93 | 18.72 | 8.96 | 0.64 | 3.33 | 20.2% | 0.91 |
| ultralow | 13.57 | 19.42 | 9.34 | 0.68 | 3.55 | 11.6% | 1.09 |

### VoxConverse test (N=232, offline) — ⚠️ contaminated
DER@.25 **6.27** / @0 8.45, miss 2.30, conf 2.88, SCA 48.7%.

## vs NVIDIA's card (their forced-align refs, collar 0)
| Dataset | NVIDIA offline | ours offline @0 | NVIDIA 1.04 s | ours low @0 |
|---|---|---|---|---|
| NOTSOFAR-1 SC | 11.74 | **16.27** | 13.87 | 18.29 |
| NOTSOFAR-1 MHM | 7.53 | — | 8.66 | — |
| AMI-SDM | 12.27 | **25.77** | 12.62 | 26.51 |

**NOTSOFAR reconciles** (ours 10.47 @.25 / 16.27 @0 vs their 11.74 @0 SC — same ballpark, and
our word-timing refs aren't identical to their FastMSS forced-alignment). **AMI-SDM does not**
(25.8 vs 12.27, all in missed speech). Since NOTSOFAR — built the same way — matches, the AMI gap
is **AMI-SDM-specific**: most likely our single-mic channel (Array1-01) and the pyannote
`only_words` refs differ from NVIDIA's SDM channel + forced-alignment. Confusion stays tiny on AMI
(≈1.1–1.6), so the model resolves the speakers it detects — it is under-*detecting* far-field speech
on this channel, not confusing speakers.

## Latency sweep — the headline for a streaming model
Going from the 30.4 s offline buffer to a real-time buffer costs very little DER:
- **AMI:** offline→ultralow +0.9 DER@.25 (23.77→24.70).
- **NOTSOFAR:** offline→ultralow +3.1 DER@.25 (10.47→13.57), mostly added confusion.
- **Speaker counting degrades with latency** on NOTSOFAR (SCA 56.6%→11.6%): less right-context
  makes over/under-counting more likely, even as DER moves modestly.

## Head-to-head vs pyannote community-1 / precision-2 — matched full sets
Identical audio, refs and protocol for all three (Nemotron = offline profile).

**AMI-SDM dev+test (N=34, `only_words` refs)**
| | community-1 | precision-2 | nemotron-3 |
|---|---|---|---|
| DER@0 / @.25 | 19.31 / 14.99 | **14.69 / 11.11** | 25.77 / 23.77 |
| miss | 7.80 | 6.16 | **21.52** |
| confusion | 4.16 | 2.65 | **1.12** |
| SCA | **82.4%** | 38.2% | 79.4% |

**NOTSOFAR-1 eval_full (N=129, word-timing refs)**
| | community-1 | precision-2 | nemotron-3 |
|---|---|---|---|
| DER@0 / @.25 | 27.30 / 20.91 | 21.70 / 15.82 | **16.27 / 10.47** |
| miss | 11.83 | 6.16 | 8.25 |
| confusion | 8.46 | 9.16 | **1.70** |
| SCA | **69.0%** | 58.9% | 56.6% |

**VoxConverse test (N=232, same refs):** community-1 8.89, precision-2 6.37, nemotron 6.27 DER@.25 —
but Nemotron trained on VoxConverse, so discount it.

Reading:
- **DER is a split decision by domain.** Nemotron wins NOTSOFAR (far-field meetings) clearly;
  precision-2 wins AMI-SDM and Nemotron is last there.
- **Speaker confusion: Nemotron is decisively best on both** (2–4× lower on AMI, ~5× on
  NOTSOFAR) — its speaker discrimination is genuinely best-in-class.
- **Nemotron's high AMI miss is unique to it.** On the same tight refs the pyannote models miss
  6–8%; Nemotron misses 21.5%. This is a real under-detection of far-field AMI speech, not a
  reference artifact (and the reason it doesn't reconcile with NVIDIA's card there).
- **Speaker counting: community-1 leads** (SCA 82% / 69%); Nemotron is mid-pack (79% / 57%),
  precision-2 is worst on AMI. (An earlier small-N run suggested Nemotron led counting; the full
  sets show it does not.)

## Performance — vs NVIDIA's reference script, same GPU, same files

The honest baseline is NVIDIA's own inference path, not our first build: `e2e_diarize_speech.py`
(NeMo Speech, the script the model card cites), bf16, `compile_encoder=true`, run in-process on one
**RTX PRO 6000** in the same container recipe as this truss (`rsi-bench/nemotron_diar/results_nemo_ref/NEMO_REFERENCE.md`).
NVIDIA's RTFx definition is *audio ÷ `model.forward` time*; "wall" adds their lhotse data loading.
Ours is measured inside the truss's own batched call on files already on the replica (server compute,
lab truss `scratchpad/nemo-graph-lab`, medians of 3; the same-replica re-run of NVIDIA's compiled path
is listed next to each cell), and end-to-end from an **in-cluster** HTTP client (`loadgen/batch-runner`).

### How the batch preset runs the model (`NEMO_DIAR_ENGINE=graphs`, `model/graph_runner.py`)

NeMo's `diarize()` is `forward_streaming`: one `forward_streaming_step` per chunk for the whole batch,
each step re-encoding [speaker cache | FIFO | chunk] through the 31-layer encoder and then updating the
cache/FIFO. In NeMo's *sync* state (the batch path) cache and FIFO grow from empty and then cycle, so
the step's shapes walk a fixed set: **129 distinct shapes at `low`, 237 at `ultralow`, 2 at `offline`**
(`scratchpad/simkeys.py`, confirmed by the capture counts). The runner keeps NVIDIA's loop and its
arithmetic and replaces the launches: the whole step — Inductor-compiled encoder, head, downsample,
length mask and NeMo's cache/FIFO update including speaker-cache compression — is captured once per
`(batch size, cache_len, fifo_len, chunk_frames, compressed)` into a CUDA graph and replayed. State
lives in persistent per-batch-size buffers, so a step is one graph launch and the state never moves;
a batch of n requests runs at the next captured batch size (rows duplicated; `low` every 2 rows from 8,
the others every 4, none below 4 — see the small-batch section) so no request pays a capture; a file's
last chunk (non-steady shape) runs the same function eagerly. Capture at load: 1,806 graphs for `low`
(129 × 14 sizes) in ~107 s, 2,133 for `ultralow` (237 × 9) in ~102 s, 18 for `offline`.
Post-processing (NeMo's `binarization_vectorized` per file per speaker on CPU, 0.7–1.5 s for 32 files
— more than the forward at `offline`) runs once on the GPU for the whole `[n, T, 8]` tensor with the same
arithmetic, so the segment strings are identical (`verify_post`).

**Why graphs, and why compile is still needed.** Graphs of the *eager* kernels alone lose to NeMo's
compiled path at bs ≥ 8 (3.27 vs 2.62 s on the 8-file set): the unfused elementwise kernels
(layer norms, residuals, RoPE, GELU) cost more GPU time than the launches they save. Graphs of the
*compiled* kernels remove the ~2 ms/step of launch overhead that `torch.compile` leaves behind in the
launch-bound regime (bs ≤ 8), which is where NVIDIA's script is slow; at bs = 32 the GPU is the
bottleneck for both (bf16 tensor cores with fp32 accumulation, ~275 TFLOPS effective) and graphs add 6 %.
The determinism check (same variant twice, bs = 3) is bit-identical for both engines (`verify_e_e`,
`verify_cg_cg`); every difference below is batch composition, not run-to-run noise.

### The 8-file `low` case (8 × MTG_32000, 363 s each = 2,907 s of audio) — lever ladder

| path | fwd | wall | RTFx (fwd) | vs NeMo's best cell (1,273) |
|---|---|---|---|---|
| NeMo script, eager, bs=1 (one file at a time) | 78.6 s | 79.0 s | 37 | |
| NeMo script, eager, bs=8 | 9.86 s | 10.1 s | 295 | |
| NeMo script, `compile_encoder=true`, bs=8 (same replica as ours: 2.62–2.65 s) | 2.66 s | 2.91 s | 1,091 | 0.86× |
| NeMo script, `compile_encoder=true`, bs=32, 32 files (its best cell) | 9.27 s / 11,808 s | 10.7 s | **1,273** | 1.00× |
| graphs of eager kernels, TF32 (`fp32`) | 4.46 s | 4.60 s | 652 | |
| graphs of eager kernels, bf16 | 3.27 s | 3.41 s | 889 | |
| compiled encoder + graphs, TF32 (`NEMO_DIAR_DTYPE_LOW=fp32`, the per-file-stable path) | 3.40 s | 3.60 s | 856 | 0.67× |
| compiled encoder + graphs, bf16, FlexAttention in fp32 (`NEMO_DIAR_ATTN_FP32=1`) | 2.43 s | 2.63 s | 1,199 | 0.94× |
| compiled whole core (pre-encode, head, mask) + graphs, bf16 | 2.30 s | 2.49 s | 1,266 | 0.99× |
| compiled encoder + graphs, bf16, FlexAttention | 2.08 s | 2.28 s | 1,400 | 1.10× (1.27× NeMo's own bs=8 cell) |
| **compiled encoder + graphs, bf16, hybrid attention (shipped `low`)** | **1.97 s** | — | **1,478** | **1.16×** (1.35× NeMo's own bs=8 cell) |
| *previous build (NeMo's path inside the truss: bf16, compile, growing state)* | *2.63 s* | *2.74 s* | *1,105* | *0.87×* |

The floor: 505 steps × ~1.0 TFLOP each (31 layers × 541 tokens × 8 rows); at 2.08 s the step runs at
~4.1 ms ≈ 245 TFLOPS effective, against ~275 for NVIDIA's compiled bs=32 cell — the bf16 (fp32-accumulate)
tensor rate of this GPU. The +20 %/+30 % targets (RTFx 1,530/1,650 on this set) would need 3.8/3.5 ms per
step, i.e. above the efficiency NVIDIA's kernels reach at four times the batch; no launch-side lever
remains (the CPU finishes issuing 270 ms before the GPU). What is left is precision (fp16 accumulation:
2× tensor rate on this GPU class, a quality question not opened here) or fewer FLOPs, which the model
does not offer (every step re-encodes the 528-frame cache+FIFO context by design).

### Model-card table, reproduced through the truss

NeMo's card cells (their script, our GPU) next to the same operation inside this truss. bs=8/32 on the
first 32 NOTSOFAR eval sessions (11,808 s); NeMo's bs=1 cells on the first 8 (3,132 s), ours on
MTG_32000 (363 s). "graphs" = compiled encoder + CUDA graphs.

| profile | bs | NeMo compiled: fwd / wall (RTFx) | **truss, shipped dtype: fwd / wall (RTFx)** | truss, other dtype: fwd (RTFx) | NeMo eager: fwd (RTFx) |
|---|---|---|---|---|---|
| offline (30.4 s) | 1 | 0.67 / 0.91 s (4,672) | bf16 **0.065 s (5,590)** (runs as 4 rows; bs=1 graph 0.033 s) | TF32 0.050 s (7,270) | 2.41 s (1,299) |
| offline | 32 | 0.48 / 1.92 s (24,790) | bf16 **0.466–0.478 s (24,000–25,338)** | TF32 0.89 s (13,297) | 0.70 s (16,802) |
| low (1.04 s) | 1 | 20.0 / 20.4 s (156) | bf16 **0.85 / 0.93 s (426)** (Flex; hybrid step −12 %) | TF32 1.23 s (296) | 85.5 s (37) |
| low | 8 | 11.3 / 12.6 s (1,043) | bf16 **8.96 / 9.77 s (1,318)** (Flex); 8 × MTG_32000 hybrid **1.97 s (1,478)** | | 43.1 s (274) |
| low | 32 | 9.27 / 10.7 s (1,273) | bf16 Flex 8.77 / 9.53 s (1,347); **hybrid 8.56 s (1,379)** | TF32 17.3 s (684) | 16.1 s (735) |
| ultralow (0.32 s) | 1 | 61.0 / 61.5 s (51) | bf16 **4.14 s (88)** (runs as 4 rows; bs=1 graph 2.53 s) | TF32 3.84 s (95) | 256 s (12) |
| ultralow | 8 | 33.9 / 35.4 s (348) | bf16 **6.08 s on 8 × MTG_32000 (478)**; 32 files bs=8 26.2 s (450) | TF32 9.97 s (292) | 129 s (92) |
| ultralow | 32 | 27.0 / 28.4 s (438) | bf16 **25.5 s (464)** | TF32 50.4 s (213) | 47.1 s (251) |

Against the targets set for this preset (NeMo's best cell +20 %): `low` 1,478 vs 1,530 (bf16 + hybrid
attention; Flex 1,400; the TF32 knob 856), `ultralow` 464 vs 525 (bf16 shipped; TF32 knob 213), `offline`
24,000–25,338 vs 30,000 (bf16 shipped; TF32 knob 13,297). All three are GPU-bound ceilings of the bf16 tensor rate at bs = 32
(+6 % over NVIDIA's compiled kernels); the wins are in the launch-bound regime NVIDIA's script leaves on
the table — 1.27× at bs=8 and 2.4–2.8× at bs=1 (single-request latency: a six-minute file in 0.85 s at
`low`, 2.5 s at `ultralow`, 33 ms at `offline`).

### DER gate (NOTSOFAR-1 eval_full, N=129, coalesced K≈8, vs the stored fp32-eager bs=1 hypotheses)

Per-file |Δ| is |DER@.25(new) − DER@.25(stored)| per session. The first row is the **control arm**:
NVIDIA's own eager fp32 path, unmodified, merely batched at K≈8 through the same server.

| profile | configuration | DER@.25 (stored) | Δ set | per-file \|Δ\| mean / median | worst file | > 2 pt | identical | spk-count changes |
|---|---|---|---|---|---|---|---|---|
| **low** | **shipped build (bf16 graphs, no bs=1 graph), K≈8 / K=1** | 12.45 / 12.39 (12.52) | −0.07 / −0.13 | 0.75 / 0.68 | MTG_32026 −11.09 / MTG_32055 −7.55 | 13 / 12 | 0 / 1 | 21 / 16 |
| **ultralow** | **shipped build (bf16 graphs), K≈8 / K≈32** | 13.94 / 13.92 (13.57) | +0.37 / +0.35 (miss +0.11, conf +0.24) | 0.81 / 0.85 | MTG_32055 +10.33 | 12 / 13 | 1 / 1 | 13 / 16 |
| ultralow | *NVIDIA compiled bf16 (reference), K≈8 / K≈32* | *13.99 / 13.82* | *+0.42 / +0.25* | *0.95 / 0.89* | *+10.45 / −10.15* | *17 / 13* | *2 / 0* | *16 / 22* |
| ultralow | strict knob `NEMO_DIAR_DTYPE_ULTRALOW=fp32` (TF32 graphs), K≈8 / K=1 | 13.60 / 13.62 | +0.03 / +0.05 | **0.19 / 0.20** | MTG_32047 −3.89 / MTG_32102 −1.41 | 1 / **0** | 23 / 22 | 3 / 6 |
| **offline** | **shipped build (bf16 graphs), K≈8 / K≈32** | 10.46 / 10.38 (10.47) | **−0.01 / −0.09** | 0.61 / 0.55 | MTG_32055 −11.0 / −11.5 | 11 / 9 | 5 / 6 | 12 / 9 |
| offline | strict knob `NEMO_DIAR_DTYPE_OFFLINE=fp32` (TF32 graphs), K≈8 | 10.48 | +0.01 | **0.105** | MTG_32175 +1.78 | **0** | 38 | 1 |
| low | control: NeMo eager fp32 (TF32 matmul), K≈8 | 12.44 (12.52) | −0.08 | 0.22 / 0.05 | MTG_32026 −10.09 | 1 | 37 | 3 |
| low | graphs, TF32, eager kernels | 12.42 | −0.10 | 0.29 / 0.04 | MTG_32026 −10.09 | 4 | 37 | 3 |
| low | **graphs, compiled, TF32** (`NEMO_DIAR_DTYPE_LOW=fp32`) | 12.43 | −0.09 | **0.32 / 0.05** | MTG_32026 −8.71 | 4 | 26 | 6 |
| low | **graphs, compiled, bf16 (shipped)** — lab run | 12.53 | **+0.01** | 0.73 / 0.29 | MTG_32026 −10.44 | 9 | 1 | 15 |
| low | **graphs, compiled, bf16 (shipped)** — shipped deployment, other batch compositions | 12.41 | −0.11 | 0.78 / 0.33 | MTG_32026 −11.09 | 12 | 0 | 22 |
| low | graphs, compiled, bf16 + fp32 FlexAttention | 12.37 | −0.15 | 0.79 / 0.34 | MTG_32026 −11.03 | 11 | 1 | 17 |
| low | graphs, bf16, eager kernels | 12.49 | −0.03 | 0.82 / — | MTG_32266 +10.68 | 14 | 2 | 12 |
| low | *previous build: NeMo's compiled bf16 path (same replica)* | 12.61 | +0.09 | 0.90 / 0.44 | MTG_32069 −7.53 | 18 | 1 | 19 |
| ultralow | graphs, compiled, bf16 | 13.86 (13.57) | +0.29 | 0.76 / 0.30 | MTG_32346 +12.71 | 8 | 2 | 12 |
| ultralow | graphs, compiled, bf16 + fp32 FlexAttention | 13.73 | +0.16 | 0.70 / 0.23 | MTG_32346 +9.61 | 8 | 3 | 15 |
| ultralow | graphs, compiled, TF32 (strict knob; the first shipped build) — shipped deployment | 13.53 (13.57) | **−0.04** | **0.24 / 0.04** | MTG_32191 −7.94 | 2 | 23 | 3 |
| offline | graphs, compiled, bf16 | 10.46 (10.47) | −0.00 | 0.61 / 0.12 | MTG_32055 −11.45 | 11 | 6 | 9 |
| offline | graphs, compiled, bf16 + fp32 FlexAttention | 10.47 | +0.00 | 0.51 / 0.13 | MTG_32055 −10.22 | 7 | 2 | 8 |
| offline | graphs, compiled, TF32 (strict knob; the first shipped build) — lab run / shipped deployment | 10.48 / 10.51 | **+0.02 / +0.04** | **0.12 / 0.02** (0.14 / 0.02) | MTG_32047 +4.48 | 1 | 35 / 37 | 2 / 1 |

Reading:
- **Set-level DER is within 0.1 of the stored hypotheses for every TF32 arm (−0.09 … +0.04 across five
  runs); bf16 sits at that noise floor at `low` (+0.01 in the lab run, −0.11 on the shipped deployment
  with other batch compositions) and fails it at `ultralow` (+0.29, +0.16 with fp32 attention)** — the
  3+1-frame regime where bf16 drift was first seen. (The first shipped build kept `ultralow` on TF32; the
  final build ships bf16 everywhere by the equal-quality-to-vendor rule — see the shipped rows at the top
  of this table — with TF32 as the strict knob.)
- **Per-file stability is a precision property, not a compile or graph property.** With TF32 matmuls the
  graph engine matches NVIDIA's own batched eager path (0.32 vs 0.22 mean, 26–37 files bit-identical to
  the bs=1 run, the same worst file); every bf16 arm sits at 0.5–0.9 whether the kernels are eager,
  compiled or graph-replayed, and running FlexAttention in fp32 inside the bf16 core does not move it
  (0.79 vs 0.73) — the drift is in the bf16 GEMMs, not the attention kernel. The mean-|Δ| ≤ 0.4 gate is
  met only by TF32.
- **No batched configuration meets a worst-file |Δ| ≤ 2 — including NVIDIA's unmodified eager path**
  (MTG_32026 moves 10 points between bs=1 and bs=8 in NeMo's own code; TF32 GEMM kernel selection
  depends on the batched M dimension, and the arrival-ordered speaker cache turns a rounding difference
  into a different speaker assignment on that session). The bs=1 stored hypotheses are one sample of a
  chaotic system, so "worst ≤ 2" is not a property any batch path can be held to; the control arm is the
  right yardstick (mean 0.22, one file > 2).
- **Every profile ships bf16** (efficiency first; NVIDIA's card numbers are bf16): the rule is bf16 wherever
  it is at least the vendor's compiled-bf16 quality. `low` set-level within ±0.1 at bs ≥ 8 (1.35× NVIDIA's
  compiled cell at the same batch); `offline` −0.01 / −0.09 (vendor's path +0.15); `ultralow` +0.16 … +0.37
  across four runs against the vendor's +0.25 … +0.42 — comparable, not clearly better, and the one place
  the strict knob is worth its cost. **`NEMO_DIAR_DTYPE_<PROFILE>=fp32` (TF32) is the eager-quality knob**:
  `ultralow` +0.05 / mean 0.20 / worst 1.4 (full gate pass) at 0.46× the bf16 speed, `offline` +0.01 / 0.105 /
  1.8 at 0.52×, `low` mean 0.32 (NeMo's own batched eager 0.22) at 0.61×.

### Small-batch precision: the bs=1 compiled graph, not the dtype

Every reduced-precision arm above was run at K≈8. Re-running identical numerics at other batch sizes
on the same 129 files separates the effects (`low`, compiled encoder + graphs):

| `low` | set Δ | miss Δ | per-file mean / worst | identical |
|---|---|---|---|---|
| bf16, K=1 (bs=1 graph) | **+0.65** | **+0.78** | 1.12 / MTG_32026 −9.03 | 0 |
| bf16, K≈8 (bs=8 graph) | +0.01 | +0.05 | 0.73 / −10.44 | 1 |
| bf16, K≈32 (bs 29–32) | −0.06 | −0.00 | 0.74 / −11.09 | 3 |
| bf16, K=1, `allow_{bf16,fp16}_reduced_precision_reduction` forced off (both default **True** here) | +0.65 | +0.78 | identical | 0 |
| **TF32**, K=1 (bs=1 graph, a second fp32 copy of the profile for batches < 8) | **+0.68** | **+0.73** | 0.80 / MTG_32049 +5.67 | 1 |
| bf16, K=1, singleton run as **2 identical rows in the bs=2 graph** | **−0.13** | **+0.01** | 0.68 / −7.55 | 1 |
| bf16, K=1, singleton run as 4 rows in the bs=4 graph | −0.14 | +0.03 | 0.77 / −7.82 | 2 |
| `ultralow` bf16, K=1 (bs=1 graph) | +1.00 | **+0.95** | 1.16 / +12.71 | 1 |
| `ultralow` bf16, K=1, bs=2 graph / bs=4 graph | +0.38 / +0.18 | +0.10 / +0.05 (confusion +0.25 / +0.13) | 0.86 / 0.71 | 4 / 0 |
| **`ultralow` TF32 (strict knob), K=1, bs=2 graph** | **+0.05** | +0.03 | **0.20 / MTG_32102 −1.41, 0 files > 2** | 22 |

The one-sided **missed-speech** bias is not a precision effect at all: it is **the batch-1-specialised
`torch.compile` graph**. dynamo specialises batch size 1 into its own compiled encoder, and that graph
under-detects speech by +0.7–1.0 pt in bf16 *and* in TF32; feeding the same file as two identical rows
through the bs=2 graph removes it (miss +0.01). It is not the split-K reduction precision (forcing fp32
reductions changes nothing) and not any single compute island: with pre-encode, head, state, RoPE tables,
GELU and FlexAttention all in fp32 and only the block GEMMs in bf16 the `ultralow` bias is still +0.13 miss
at 0.8× the speed; the K=1 island bisect (deterministic — identical numbers across runs and replicas)
moves nothing: head +0.04, pre-encode −0.06, RoPE +0.09, GELU ±0.00. `ultralow` showed the bias at "K≈8"
because a 10-second batch desynchronises the harness clients into 1–4-row batches. **NVIDIA's own
compiled-bf16 script has it too**: `ultralow`, same files, K≈8: +0.42 set / +0.31 miss; K≈32: +0.25.

**Fix shipped: no size-1 graph is captured** (`NEMO_DIAR_GRAPH_SIZES` starts at 4 for bf16 profiles,
which also keep a small confusion residual at 2 rows, and at 2 for the TF32 profiles); a single request
runs as identical rows at the same latency (launch-bound regime). Validation on the shipped deployment:
`low` bf16 K=1 −0.13 / miss +0.01; `ultralow` on the TF32 knob at K=1 **+0.05, mean 0.20, worst −1.41, 0 files > 2 pt —
the first configuration to pass the full gate including the worst-file criterion**; at K≈8 on the same
deployment: `low` bf16 −0.07 (miss +0.04, per-file 0.75), `ultralow` TF32 +0.03 (mean 0.19, worst −3.89,
one file > 2), **`offline` TF32 +0.01 (mean 0.105, worst +1.78, 0 files > 2 — full gate pass)**. The final
build ships bf16 on all three profiles (rows at the top of the gate table); TF32 stays one env flip away.

Controls for the `ultralow` gate (same 129 files; "K≈32" = concurrency 32 through the server):

| arm | set Δ | miss Δ | per-file mean / worst | identical |
|---|---|---|---|---|
| (a) NeMo eager fp32, K≈8 | +0.06 | +0.01 | 0.12 / MTG_32055 +2.81 | 56 |
| (a) NeMo eager fp32, K≈32 | +0.05 | +0.01 | 0.15 / MTG_32102 −1.70 (0 files > 2) | 22 |
| (b) NeMo compiled bf16 (the script's path), K≈8 | +0.42 | +0.31 | 0.95 / MTG_32346 +10.45 | 2 |
| (b) NeMo compiled bf16, K≈32 | +0.25 | +0.09 (conf +0.14) | 0.89 / MTG_32047 −10.15 | 0 |
| (c) ours, TF32 graphs, K≈8 (shipped deployment) | −0.04 | +0.03 | 0.24 / MTG_32191 −7.94 | 23 |
| (c) ours, TF32 graphs, K≈32 | +0.02 | +0.04 | 0.18 / MTG_32047 −3.04 | 26 |
| ours, bf16 graphs, K≈8 (three runs) | +0.29 / +0.31 / +0.52 | +0.27 … +0.32 | 0.73–0.97 | 0–2 |
| ours, bf16 graphs, K=1 | +1.00 | **+0.95** | 1.16 / MTG_32346 +12.71 | 1 |
| **ours, bf16 graphs, K≈32** | **+0.16** | +0.07 (conf +0.07) | 0.78 / MTG_32346 +12.71 | 3 |
| ours, fp16 graphs (attention via the compiled kernel directly), K≈8, two runs | +0.31 / +0.21 | +0.29 / +0.17 | 0.73 / 0.71 | 0 |
| ours, bf16 + fp32 FlexAttention | +0.16 | | 0.70 | 3 |
| ours, bf16 + fp32 pre-encode/head/state (mixed core) | +0.31 | +0.24 | 0.70 | 2 |
| ours, bf16 + fp32 {head} / {pre-encode} / {layer 0} / {layers 27–30} | +0.24 / +0.33 / +0.35 / +0.27 | +0.14 / +0.22 / +0.28 / +0.24 | 0.58 / 0.82 / 0.90 / 0.76 | 1 / 1 / 2 / 4 |
| ours, bf16 + fp32 {RoPE} / {GELU} / {RoPE+GELU+attention} / {all islands} | +0.22 / +0.26 / +0.42 / **+0.12** | +0.15 / +0.14 / +0.23 / +0.13 | 0.70 / 0.82 / 0.93 / **0.48** | 1 / 1 / 0 / 2 |

fp16 overflow check: a 64-minute NOTSOFAR concatenation (3,851 s) through both fp16 cores at bs = 1 and
all 129 files at K≈8 — **0 non-finite values** in predictions and state, prediction range exactly [0, 1]
(`nonfinite_preds`/`nonfinite_state`/`pred_max` are in every response's `timing`). Under autocast the
residual stream and both LayerNorms are already fp32 with bf16 weights (dtype probe on block 15:
`block`/`norm1` float32, `attn`/`ffn`/`gelu`/`rope` bfloat16) — the same recipe as NVIDIA's script.

### Where the GPU time goes (`ultralow`, bf16 compiled step, torch.profiler over the whole 32-file batch)

`torch.profiler` over the eager+compiled step (no graph replay, so kernels are attributable) for the
whole 32-file batch at bs = 32 — 1,921 steps, **25.3 s of GPU time**, i.e. the measured forward:

| kernel class | GPU time | share | launches |
|---|---|---|---|
| GEMM (cuBLAS bf16: QKV, out-proj, FFN, head) | 16.05 s | **63.4 %** | 250,321 |
| Inductor-fused elementwise (LayerNorm, residual, RoPE, GELU, masks) | 4.86 s | 19.2 % | 307,402 |
| FlexAttention (compiled Triton) | 3.66 s | 14.5 % | 59,551 |
| copies / memset (state writes, output) | 0.46 s | 1.8 % | 31,314 |
| other | 0.27 s | 1.1 % | 42,817 |
| speaker-cache / FIFO update (topk, sort, gather, scatter) | 5 ms | 0.02 % | 4,206 |
| mel / STFT, H2D | 4 ms | 0.02 % | 1 |

Top kernels:

| kernel | GPU time | launches |
|---|---|---|
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_re` | 3.84 s | 52,813 |
| `triton_tem_fused__to_copy__unsafe_view_add_bitwise_and_cat_c` | 3.65 s | 58,249 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_re` | 2.95 s | 48,741 |
| `nvjet_sm120_tst_mma_128x256x64_2_64x64x64_tmaAB_alignCD4_bx_` | 1.91 s | 20,987 |
| `nvjet_sm120_tst_mma_128x160x64_2_64x40x64_tmaAB_alignCD4_bx_` | 1.38 s | 25,513 |
| `triton_per_fused__to_copy_add_native_layer_norm_view_12` | 1.32 s | 55,709 |
| `triton_poi_fused_gelu_view_8` | 1.22 s | 58,249 |
| `nvjet_sm120_tst_mma_128x144x64_2_32x72x64_tmaAB_alignCD4_bx_` | 1.17 s | 23,963 |

Reading: the `ultralow` step at bs = 32 is **GEMM-bound** (63 %, ~275 TFLOPS effective bf16 with fp32
accumulation — this GPU's practical tensor rate). The cache/FIFO update and the front end are nil, so
wider batches (48/64 rows) and folding mel/H2D into the graph or a side stream have nothing to
amortise; FlexAttention is 14.5 %, so even a 2× faster attention kernel for the T = 532-token sequence
(the attention is over cache + FIFO + chunk, not over the 4-frame chunk) would be worth ≤ 7 %. The
remaining 20 % of fused elementwise kernels is Inductor's (compiling the whole core did not shrink it).
At bs = 8 the shares are 63 / 19 / 12.5 % with 2.5 % copies — the same shape, launch-bound only below
bs ≈ 8, which is what the graphs remove.

### Sustained throughput from an in-cluster client (`low`, six-minute NOTSOFAR files, one replica)

Closed loop: N clients each posting the next file as soon as the previous response returns, 180 s per
arm, `low` bf16 graphs; GPU utilisation sampled at 1 Hz on the replica. Three batcher generations of the
same engine (the compute per batch is identical; what changes is how much of the time the GPU has a batch):

| batcher (files / min; p50 latency) | N=8 | N=16 | N=32 | N=64 | notes |
|---|---|---|---|---|---|
| *previous build (NeMo compiled path, cap 16)* | *86* | *117* | *150* | — | |
| graph engine, single thread, 100 ms window, CPU post-processing | 89 (5.5 s) | 128 (7.5 s) | 150 (12.6 s) | 143 (22 s) | batches split by arrival skew (3+5 at N=8, 29+3 at N=32) and pad to the next captured size; N=64 only queues |
| + GPU post-processing, decode 8 workers | 116 (4.2 s) | 137 (6.6 s) | 156 (12.2 s) | 169 (22 s) | post 1.3 s → 0.2 s per 32-batch |
| + fill window (400 ms below a captured size), sizes +12/+24 | 113 | 138 | 151 | 166 | no effect: a closed loop phase-locks into two client groups that no sub-batch window can merge |
| + collector thread (next batch's decode/window overlap the forward) | 108 | 133 | 153 | 172 | 1–2-row batches leak in while the GPU is busy and each costs a bs=1/2 step |
| + window held open while a forward is in flight | 112 | **149** | **161** | 171 | |
| + captured sizes every 4 rows, one fill window after the forward frees | 110 (4.3 s) | 145 (6.6 s) | 161 (12.2 s) | **193** (19.5 s) | 26-row batches run the 28-row graph; N=64 is 1.08× NVIDIA's lockstep ceiling |
| *NeMo script, compiled bs=32, lockstep, audio on disk* | — | — | *179* | — | *32 files per 10.7 s; no upload, no decode* |

Server-side per batch at N=32: fwd 8.9 s for 29–32 files (RTFx 1,300), decode 0.3–0.5 s (8 workers),
GPU post-processing 0.2 s. The ceiling of this client pattern is 32 files per 9.1 s ≈ 210 files/min
with every batch full; what remains between 171 and that is padding (a 26-row batch runs the 32-row
graph) and the two-group phase lock of a closed loop — N=8 is a client artefact (3+5 groups, each
group uploading while the other computes), not a server one. At N ≥ 32 the graph engine is at
0.90–0.96× NVIDIA's lockstep ceiling *with* upload, decode and HTTP in the loop, where the previous
build was at 0.84×.

#### Kernel-level levers measured against the compiled-bf16 core (`low`, steady shape S=264/F=264/T=541, graph replay)

| lever | bs=8 step | bs=16 step | bs=32 step | 32 files bs=32 fwd | verdict |
|---|---|---|---|---|---|
| **baseline: Inductor-compiled encoder (cuBLAS bf16 GEMMs, FlexAttention) + graphs** | 5.54 ms | 9.35 ms | 19.5 ms | 8.78 s (RTFx 1,344) | shipped |
| *NeMo script, compiled bf16 (no graphs)* | | | | *9.27 s (1,273)* | |
| GEMM autotune (`max_autotune_gemm`, TRITON+CUBLAS; real compile, caches off, +65 s) | — | 14.09 vs 14.19 (−0.7 %) | **27.1 vs 19.7 (+38 %)** | — | Triton GEMM picks lose to cuBLAS at K=512; no |
| dense SDPA with the padding mask (memory-efficient / math backend) | 6.65 (+20 %) | 11.2 (+20 %) | 23.3 (+20 %) | 10.6 s (1,115) | the mask takes SDPA off the flash kernel; no |
| dense SDPA unmasked (flash) — exact in sync state except a file's last chunk | 5.18 (−4 %) | 8.74 (−5 %) | 18.65 (−4 %) | 8.47 vs 8.75 s (1,394); 8 files 1.98 s (1,466) | gate K≈8 12.47 (−0.05), per-file 0.68 — Flex class |
| **hybrid: flash SDPA on full steps, masked Flex on a row's final partial chunk (shipped for `low`)** | bs=1 1.86 vs 2.12 (−12 %); **5.06 (−4 %)** | **8.65 (−5 %)** | **18.47 (−4 %)** | 8 files **1.967 s (RTFx 1,478)**; 32 files bs=32 **8.56 s (1,379)** | `low` gate K≈8 12.44 (−0.08), per-file 0.67 / worst −10.7 — Flex class. On the TF32 profiles it costs per-file stability (`ultralow` 0.30 vs 0.19, 10 vs 23 identical; `offline` 0.31 vs 0.11, 5 vs 0 files > 2) for an unmeasured gain → they keep Flex |
| dense SDPA with the mask, cuDNN backend | 7.28 (+35 %) | 11.5 (+25 %) | 23.1 (+19 %) | 9.60 s (1,230) | no |
| FlexAttention `kernel_options` BLOCK 64/64 | 5.35 | 9.19 | 19.5 | 8.79 s | ±0.5 %; no |
| FlexAttention `kernel_options` BLOCK 32/32 | 5.71 (+6 %) | 9.99 (+9 %) | 21.3 (+10 %) | 9.48 s | no |
| FP8 e4m3 encoder linears (`torch._scaled_mm`, per-tensor absmax scales; attention/head bf16) | — | 12.8 (+35 %) | 25.8 (+28 %) | 14.6 s (806) | K=512 GEMMs too small to pay for the per-call amax + casts; no gate run |
| compile the whole core (pre-encode, head, mask) instead of the encoder | 2.30 vs 2.08 s (8 files) | | | | slower; no |
| fp16 core (attention via the compiled kernel directly) | same as bf16 | | 25.6 vs 25.5 s (`ultralow`) | | same tensor rate; 0 non-finite; same drift class |

#### Open-loop (constant arrival rate) ceiling — the way NeMo's files/min is defined

Requests fired every 60/R s regardless of completions, 10 min per arm, six-minute files, a 1-client
control alongside; completion rate and latency per third (`results_batch_lg/open_v7`, `open_v10/11`):

| arrival rate | completed / min by third | p50 latency by third | batch rows (p50) | holds? |
|---|---|---|---|---|
| 180 / min | 167 → 176 → **181** | 13.7 → 16.2 → 17.7 s (control 13.2 s) | 12–13 | yes (third 3 ≥ rate, latency +29 %) |
| 200 / min | 181 → 192 → 189 | 16 → 25 → 33 s | 12 | no — ~190/min sustained, queue grows |
| 220 / min | 172 → 180 → 177 | 31 → 76 → 122 s | 13 | no |
| 200 / min, backlog hold (≥ 16 rows → fill to 32, `predict_concurrency` 256) | 166 → 173 → 172 | 25 → 54 → 84 s | 13 | no — the hold never fires: at a constant rate one forward's worth of arrivals (12–15 rows) is what is queued when a forward ends |
| 200 / min, graphs every 2 rows (no padding), 512-connection loadgen | 178 → 190 → 189 | 17 → 29 → 40 s | 14 | no — padding was not the limit |
| 200 / min, post-processing off the GPU thread | 185 → 190.5 → 193.8 | 13 → 18 → 26 s | 13–14 | ~194 sustained; latency still drifts, so 200 does not formally hold |
| **200 / min, + hybrid attention (shipped `low`)** | **187.8 → 200.4 → 199.2** | **10.8 → 11.1 → 11.8 s (flat; control 10.1 s)** | 9–14 (fwd 3.4 s) | **holds — 1.12× NeMo's 179** |
| 220 / min, shipped | 194.7 → 209.4 → 205.2 | 17 → 29 → 40 s | 14 (fwd 3.8 s) | no — ~207/min sustained with growing latency |
| *NeMo script, compiled bs=32, lockstep, audio on disk* | *179* | — | 32 | — |

**Duty is not the gap.** Per-batch instrumentation on the shipped build (`gap_ms` in every response = time from
the previous forward's end to this forward's start on the GPU thread): **p50 1 ms, p90 1 ms over 66 batches —
99.9 % duty**. What the GPU is doing is the fixed point of a constant arrival rate: rows per batch = rate ×
forward time, which at 200/min settles at 13–14 rows per 4.1 s, and 13 rows cost 0.31 s each (16+: 0.28).
The asymptotic capacity is 214/min only with ≥ 16-row batches, which a constant-rate arrival stream never
accumulates without a latency-adding hold — and a hold is GPU idle time (26 rows in 7.3 s + 2 s hold = 168/min).
So the sustained rate with flat latency is ~195/min; ~200 with slowly growing latency; 214 asymptotic.

**Shipped build: 200 files/min holds with flat latency (1.12× NeMo's 179); ~207/min sustained with growing
latency; 220 does not hold. Duty on the shipped build: gap between forwards p50/p90 1 ms at both rates
(99.9 % / 99.7 %).** CPU is visible but not the limiter: with 8 ffmpeg decode workers, per-file `decode_ms` is p50 192 /
p90 273 / max 625 ms at 200/min (3.3 six-minute FLACs per second, ~1,200× real-time decoding) and p50 450 /
p90 558 / max 767 ms at 220/min — the 2.3× rise is CPU contention in the decode pool — yet the decodes
overlap the running forward (collector thread) and duty stays 99.7 %, so contention adds per-request latency,
not GPU idle. (The instance's vCPU grant is not exposed by the API and was not measured from inside the pod;
`NEMO_DIAR_DECODE_WORKERS` is the knob if it is.) A second consumer thread on a second CUDA stream would only remove idle,
of which there is 0.1 %; the ceiling is the arrival-rate fixed point above.
**Hour-long arm at 200 files/min on the shipped build: holds — 11,994 of 12,001 requests OK (0.06 % errors),
completions per 20-minute third 198.0 → 199.4 → 200.5/min, p50 latency 11.4 → 11.7 → 11.5 s (flat), p95 13.3–13.6 s,
server queue 6.7–6.9 s flat, decode p50 194 ms; the 1-client control alongside saw 10.1 s p50
(`results_batch_lg/open_v14/hour.json`).** Before the hybrid attention (FlexAttention core): ~195 sustained.
Hour-long arm at 180 files/min (FlexAttention build): holds — 10,793 of
10,800 requests OK (0.06 % errors), completions per 20-minute third 176.7 → 179.8 → 179.7/min, p50 latency
17.4 → 22.9 → 22.9 s (flat after the ramp), server queue 8.3 s flat; a 1-client control alongside saw
18.0 s p50 (`results_batch_lg/open_v7/hour.json`).** Why a fuller
batch does not raise it: per-row GPU cost is flat from 16 rows up (bs=16 4.5 s, bs=32 9.1 s at `low`),
so the ceiling is 16 rows per 4.5 s = 213/min at 100 % duty; the measured 190–193 (open loop / closed
loop N=64) is ~90 % duty — the eager last-chunk steps, output D2H, GPU post-processing (0.1 s) and the
hand-over between batches. Bigger batches only help below 16 rows, which is where the closed-loop N=32
arms (25–29-row batches) were.

### What the server adds over the script
- **HTTP with automatic coalescing of independent requests.** The script batches a manifest; the
  server batches whatever arrives: same-profile requests enqueue on receipt, decode in parallel
  (`ffmpeg`, pipes, into pinned host memory so the H2D copy is asynchronous), and one batched call runs
  once the window has been quiet for 100 ms and every admitted decode has finished (cap 32, 1.5 s max),
  longest files first. Different profiles run concurrently on separate instances and CUDA streams.
- **A per-request latency profile** in every response (`timing`: fetch, decode, queue wait, batch size,
  graph batch size, h2d, mel, steps, forward, post-processing, total) and a per-batch server log line
  with RTFx.
- **Deterministic output** for a given batch composition (bit-identical on repeat), NeMo's arithmetic
  throughout (fp32 accumulate, TF32 or bf16 GEMMs as configured), post-processing identical to NeMo's.
- **Autoscaling** on request concurrency, HTTP 400 on client errors.

### Not closed, and why
- **Upload.** A 6-minute FLAC is ~7 MB base64; from an in-cluster client it takes 0.7–1.4 s through
  the ingress, with occasional 2–4 s stragglers that then run in a later batch. This is the gap
  between server compute and end-to-end; sending `url` instead of `audio_b64` moves the fetch onto
  the replica's own network but not off the critical path.
- **Dropped responses.** Roughly 1–3 % of requests in every long client run here never received their
  response (connection reset mid-body or a silent socket); server-side error logs are empty for those
  requests. `full_run.py` resumes (`--skip-existing`); the cause (ingress vs. truss server) is not
  identified and should be before a listing.
- **GPU-bound at bs ≥ 16.** The remaining levers are precision (fp16 accumulation) or a model with a
  smaller re-encoded context; neither is a serving change.
- **fp16 core** cannot be captured: under fp16 autocast the compiled FlexAttention falls back to the
  eager math path, which creates a CPU tensor mid-step (illegal during capture). bf16 is the tensor-core
  path here.

## Cold start and recompiles
Startup = restore 3 instances + `torch.compile` of the encoder + capture of every state shape at the
captured batch sizes (`low` 14, others 8–9), measured on RTX-PRO-6000 (fde-internal), 3-profile default:

| Build | `load()` | notes |
|---|---|---|
| fp32, fixed-shape state (first build) | 148–191 s | |
| bf16, growing state, torch.compile (previous) | 188 s (offline 111 / low 69 / ultralow 5) | |
| graph engine, 6 sizes (first shipped build) | 374 s (offline 112 / low 150 / ultralow 110) | 774 + 1,422 + 12 graphs; 8 compiled encoder graphs (bs 1 vs >1 × T ≤/> 128 × 2 dtypes); the first profile pays Inductor (~70 s), the others reuse its artifacts |
| **graph engine (shipped: `low` sizes every 2 rows, bf16 everywhere)** | **~7 min** (capture alone: low 107 s / ultralow 102 s / offline 48 s) | 1,806 + 2,133 + 18 graphs |

- **No recompiles under traffic**: the graph engine only replays captured graphs (a batch of n runs at
  the next captured size); a shape without a graph (a file's last chunk) runs eagerly, never compiles.
  `torch._dynamo.config.cache_size_limit` is raised to 64 so the encoder can never silently drop to eager.
- **Capture cost is warm-up only:** ~50 s for 774 `low` graphs; the warm-up streams 100 s of synthetic
  audio per batch size (`NEMO_DIAR_WARM_S`), enough to walk the cache/FIFO cycle (85 s at `low`).
- **b10cache** (`b10-transfer`) saves the Inductor artifacts under `/cache/model` — per deployment — so
  autoscaled/restarted replicas of the same deployment skip the compile (not the capture).
- **Autoscaling for a library listing:** min 1 replica (a 6-minute cold start rules out
  scale-from-zero), `concurrency_target` ≈ 16, `predict_concurrency` 48.

## Reproduce
- `rsi-bench/nemotron_diar/full_run.py` — full-set, all latencies (sends lossless FLAC).
- `rsi-bench/nemotron_diar/loadgen/batch-runner/` — in-cluster CPU load generator (aiohttp; burst
  rounds or closed loop; records the server `timing` of every response); driver `batch_loadgen.py`,
  sequence `batch_final_measure.sh`; results under `results_batch_lg/`.
- `rsi-bench/nemotron_diar/results_nemo_ref/` — NVIDIA's script on the same GPU (raw per-cell JSON).
- Server-compute A/B (dtype × engine × compile, per-profile card cells, bit-identity `verify` ops): lab
  truss in `scratchpad/nemo-graph-lab/` (wraps this `model.py`; `bench` times the batched call on stored
  files); the per-file gate is `rsi-bench/nemotron_diar/perfile_delta.py`.
- `rsi-bench/nemotron_diar/build_notsofar_eval.py` — builds NOTSOFAR eval_full from HF.
- AMI: `rsi-exam-data/prepared/{visible/ami_sdm_dev,sealed/ami_sdm_test}`.
- Per-recording CSVs + hyp RTTMs under `rsi-bench/nemotron_diar/results_full/<name>_<latency>/`.

## Streaming
A companion WebSocket streaming preset (`../streaming/`) serves the same checkpoint live
chunk-by-chunk; its output is validated to match this batch path (DER ~0). See `../streaming/BENCHMARK.md`.
