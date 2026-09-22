// Stream a 16 kHz mono PCM16 WAV to the Nemotron 3 Diarization streaming endpoint.
//   export BASETEN_API_KEY=... MODEL_ID=...
//   npm i ws && node client.js meeting.wav [low|ultralow|offline]
// Each server frame carries the full current turn list (replace-style).
const fs = require("fs");
const WebSocket = require("ws");

const [, , wavPath, latency = "low"] = process.argv;
if (!wavPath) { console.error("usage: node client.js meeting.wav [latency]"); process.exit(2); }

const FRAME_BYTES = 3200; // 100 ms of PCM16 @ 16 kHz
const pcm = fs.readFileSync(wavPath).subarray(44); // skip the canonical 44-byte WAV header
const url = `wss://model-${process.env.MODEL_ID}.api.baseten.co/environments/production/websocket`;
const ws = new WebSocket(url, { headers: { Authorization: `Api-Key ${process.env.BASETEN_API_KEY}` } });

ws.on("open", () => {
  ws.send(JSON.stringify({ latency }));
  let off = 0;
  const tick = setInterval(() => {
    if (off >= pcm.length) {
      clearInterval(tick);
      ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
      return;
    }
    const audio = pcm.subarray(off, off + FRAME_BYTES).toString("base64");
    ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio }));
    off += FRAME_BYTES;
  }, 100); // real-time pacing
});

ws.on("message", (data) => {
  const frame = JSON.parse(data);
  if (frame.type === "error") { console.error("server error:", frame.error); process.exit(1); }
  const tail = frame.turns.slice(-2).map((t) => `${t.speaker}[${t.start.toFixed(1)}-${t.end.toFixed(1)}]`);
  console.log(`t=${frame.processed_s.toFixed(1)}s speakers=${frame.num_speakers} ${tail.join("  ")}`);
  if (frame.is_final) {
    console.log(`\nfinal: ${frame.num_speakers} speaker(s), ${frame.turns.length} turns`);
    for (const t of frame.turns) console.log(`${t.start.toFixed(2)}\t${t.end.toFixed(2)}\t${t.speaker}`);
    ws.close();
  }
});
ws.on("error", (e) => { console.error(e); process.exit(1); });
