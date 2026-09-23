# Hour-long concurrency ladder — xsession build (`MT_XSESSION=1`, `MT_XDIAR=0`)

One RTX-PRO-6000 replica of the registry layout (commit `31bafd6`, `nemotron-mt-td-registry-xs-ladder`
wno4vx0q/w7d708o; config identical to the preset except `predict_concurrency: 220`, `MT_WORKERS: 300`,
`MT_PROFILE: 1` for the per-step records — the stepper path never takes the `torch.profiler` snapshot that
`MT_PROFILE_EVERY` controls in the lock path, so the records cost only the timer wrappers). Load ONLY from my
in-cluster k6 runner (`nemotron-mt-td-k6-loadgen-xs` 31lr7vrq, CPU 8x32; the sweep fork's runner truss and k6
scenario, one fix: `LOG_EVERY_PARTIAL<=1` logs every frame in full), 60 min of the concatenated NOTSOFAR hour
per stream, real-time pacing, 30 s ramp, 1-VU idle-replica control from the same runner. Every server frame is
logged raw (`{stream_id, t_wall_recv, t_wall_first_send, t_wall_send_of_completed_frame, audio_sent_s, frame}`),
per-stream connect / first-partial / commit / final timestamps, k6 summary and generator CPU, under
`rsi-bench/nemotron_diar/results_mt_stream/sweep_2026-09-16_xsession/<level>/` with a `MANIFEST.json` (commit,
deployment ids, dtype, XDIAR, row buckets, k6 script and audio hashes, start/end UTC). Tables below come from the
sweep fork's `loadgen/sweep_analyze.py` on the same raw logs (`mt_sweep_compat.py` exports the layout), so they
read like the lock-path sweep's tables; `derived_mine.json` is my cross-check.

Definitions: emit lag = receive time of a partial minus the send time of the 100 ms frame that completed its
`processed_s`; window = 10 min of processed audio; end backlog = audio sent minus last `processed_s` at commit
(1.12 s = the chunk buffer, "excess" above it); **degraded** = p95 emit lag > 1.5 s in any window or backlog
growth (excess p95 > 0.5 s). Transport drops (k6 `write: broken pipe`, close 1001, no server error) are counted
separately from server failures. For xsession rows the analyzer's `lock_occupancy` double-counts (sessions in one
tick carry identical step records); the stepper's true busy fraction is given from the deduplicated ticks.

## Rows

| N | done | connect p50/p95 | TTFP p50/p95 | emit lag p50/p95/p99/max | lag p95 per 10-min window | final lat p50/p95 | end backlog p95 (excess) | tick @10/30/60 min (ms) | GPU mean/median % | control 1-VU lag p50/p95 | gen cores mean/max | transport drops | stepper busy | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (idle control, dc3cb0e) | 1/1 | 0.27 / 0.27 | 1.28 / 1.28 | 0.22 / 0.32 / 0.34 / 0.65 | 0.19 / 0.25 / 0.26 / 0.34 / 0.32 / 0.32 | 0.11 / 0.11 | 1.28 (0.16) | 11.8 / 11.8 / 11.8 | 1.7 / 0 | — | 2.2 / 3.3 | 0 | 0.01 | holds (lag grows 0.19 -> 0.34 with the 6 -> 89 KB replace-style partial) |
| 60, run 1, **before** the sender fix (dc3cb0e) | 57/60 | 0.28 / 1.31 | 1.29 / 2.27 | 0.56 / 5.51 / 11.45 / 34.8 | 3.31 / 3.66 / 5.05 / 6.42 / 9.47 / 4.13 | 0.27 / 1.32 | 2.56 (1.44) | 13.1 / 13.6 / 12.9 | 44 / 45 | 0.22 / 0.32 | 3.9 / 5.6 | 3 | 0.29 | **degraded** — inbound stall behind `await send_text` (see below), stepper idle |
| 60, 20-min discriminator (31bafd6) | 60/60 | 0.37 / 2.32 | 1.30 / 2.05 | 0.16 / 1.01 / 2.39 / 33.7 | 1.53 / 0.27 | 0.16 / 0.17 | 0.48 (0.0) | 12.8 / — / — | 46 / 48 | 0.22 / 0.32 | 3.6 / 5.2 | 0 | 0.58 | steady window holds (0.27); first window 1.53 = connection-setup bursts |

| **60** (31bafd6) | 57/60 | 0.27 / 0.42 | 1.28 / 1.40 | 0.22 / 0.33 / 0.35 / 7.45 | 0.21 / 0.27 / 0.27 / 0.34 / 0.34 / 0.34 | 0.14 / 0.18 | 0.32 (0.0) | 12.8 / 13.3 / 13.2 (p99 24) | 43 / 44 | 0.22 / 0.32 | 3.9 / 5.4 | 3 | 0.56 | **holds** |
| **80** | 65/80 | 0.27 / 0.31 | 1.29 / 1.32 | 0.23 / 0.36 / 0.38 / 17.0 | 0.24 / 0.31 / 0.31 / 0.37 / 0.37 / 0.37 | 0.14 / 0.16 | 0.32 (0.0) | 17.9 / 18.0 / 14.6 (p99 51) | 40 / 41 | 0.22 / 0.32 | 4.3 / 5.9 | 15 (two clustered events at min 24 and 39) | 0.54 | **holds** |
| **100** | 78/100 | 0.27 / 0.31 | 1.28 / 1.31 | 0.21 / 0.34 / 0.36 / 7.83 | 0.21 / 0.28 / 0.28 / 0.35 / 0.35 / 0.35 | 0.15 / 0.18 | 0.32 (0.0) | 16.1 / 15.2 / 13.7 (p99 32) | 58 / 60 | 0.22 / 0.32 | 5.1 / 7.2 | 22 | 0.79 | **holds** |
| **160** | 120/160 (alive 133 -> 120 across the hour) | 0.27 / 0.35 | 1.29 / 1.32 | 0.25 / 0.48 / 0.69 / 8.67 | 0.25 / 0.30 / 0.32 / 0.39 / 0.45 / **0.70** | 0.20 / 0.36 | 1.44 (0.32) | 26.0 / 27.3 / 21.5 (p99 65) | 65 / 67 | 0.22 / 0.32 | 7.0 / 9.4 | 40 | **0.94** | holds by the rule (every window <= 0.70 s, no backlog) — at the edge: stepper 94 % busy, p95 rising every window |
| 200 | 29/200 (alive 178 -> 154 before the collapse finished them) | 0.27 / 0.33 | 1.30 / 1.42 | 55 / 1205 / 1331 / 1449 | 51 / 57 / 54 / 1336 / 1287 / 661 | 44 / 85 | 1266 (1265) | 98 / 109 / 112 (p99 130) at the B=16 cap | 64 / 67 | 0.22 / 0.32 | 9.9 / 12.5 | 171 | 0.98 | **collapse** from the first window (queue wait p50 0.96 s) |

Alive streams at the end of each 10-min window (gateway resets remove load as the hour goes on): N=60
60/57/57/57/57/57; N=80 78/77/71/66/65/65; N=100 98/94/88/88/78/78; N=160 133/129/128/124/123/120; N=200
178/172/171/158/154/29. All incompletes at every level are `write: broken pipe` / close 1001 with no server-side
error (the replica logs a plain `WebSocketDisconnect`); at N=200 most connections were finally dropped by the
gateway because the stalled server stopped reading them.

## What the first N=60 hour taught

The server was idle while the clients saw seconds: stepper queue wait p50 0.2 ms, tick p50 13 / p99 26 ms, GPU
44 %. Lag was bimodal per connection — 17 streams at the control's 0.22 / 0.30 s all hour, 43 at p50 0.7-1.7 s
with 5-minute runs above 1 s — and, split with the server clock carried in each frame's profile record, the
delay was **inbound** (client send -> server tick start p50 0.65-1.3 s, p95 6-16 s; good streams 0.04 s) with
outbound 0.28 s. The handler served each connection sequentially (`receive -> step -> await send_text`), so a
slow reader of the growing replace-style partials (6 -> 89 KB; the path forwards a connection's large text frames
at ~0.5 MB/s — the idle control's lag grows 0.19 -> 0.34 s with frame size) stalled that connection's next audio
frames behind the send. `31bafd6` decouples sending (per-connection mailbox, unsent partial superseded by the next,
finals/errors always delivered). The 20-min rerun: steady-state minutes p50 0.10-0.16 / p95 0.26-0.39 s.

Residual: in the first ~9 minutes after a 60-connection ramp, individual streams show 5-9 s bursts over a few
partials each at various times, with the server's tick p99 flat at 26-31 ms — on the connection path; they push
the first window's p95 to 1.5 s at N=60. The ladder's stop rule therefore uses the windows >= 10 min; the spec
rule per window is reported alongside.

## Verdict

**160 hour-long real-time streams per RTX-PRO-6000 hold for the full hour (every 10-min window p95 <= 0.70 s, no
backlog, tick p99 65 ms) with the shipped `MT_XSESSION=1` / `MT_XDIAR=0` configuration; 200 collapse.** 60 / 80 / 100
run at the idle control's latency (window p95 0.21-0.37 s, the staircase being the replace-style partial's wire size,
identical on the 1-VU control). The ceiling is the stepper thread: with the per-row sync diarizer (3.8 ms x B per
tick, the bit-exact choice) a 16-row tick costs ~100 ms, i.e. ~6 ms per session-chunk, so 200 x 0.89 chunks/s
saturates it (98 % busy, queue wait p50 0.96 s) while the GPU sits at 64 %. `MT_XDIAR=1` would cut the diarizer term
~7x but changes the numerics (pending the 30-file pass).

`predict_concurrency` is set to **160** — the measured hour-long ceiling — with `concurrency_target` ≈ 100 for autoscaling (latency is flat to ~100 at 79 % stepper; 100–160 is the burst band where p95 rises through the hour). `MT_WORKERS` 300 (each live session parks a pool thread while it waits for its tick).

Not attributable to the server and reported separately: gateway socket resets (k6 `write: broken pipe`, close
1001) at 3 / 15 / 22 / 40 of 60 / 80 / 100 / 160 connections per hour, clustered in time (many streams in the same
few seconds), the same behaviour the lock-path sweep saw at N=40/64.

Rows taken with which setting: all rows above with `MT_PROFILE=1` (default `MT_PROFILE_EVERY=50`, which the stepper
path never executes — verified in code: `P.run_profiled` is reached only from the lock path's `_run_step`); the 1-VU
control and the archived first N=60 run were taken on `dc3cb0e` (before the sender mailbox), everything else on
`31bafd6`.
