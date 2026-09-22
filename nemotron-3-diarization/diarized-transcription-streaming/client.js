// Stream a 16 kHz mono PCM16 WAV to the Nemotron 3 diarized-transcription endpoint.
//   export BASETEN_API_KEY=... MODEL_ID=...
//   npm i ws && node client.js meeting.wav [maxSpeakers]
// Prints each closed speaker-tagged turn once, plus each speaker's live partial.
const fs = require("fs");
const WebSocket = require("ws");

const [, , wavPath, maxSpeakers = "8"] = process.argv;
if (!wavPath) { console.error("usage: node client.js meeting.wav [maxSpeakers]"); process.exit(2); }

const FRAME_BYTES = 3200; // 100 ms of PCM16 @ 16 kHz
const pcm = fs.readFileSync(wavPath).subarray(44); // skip the canonical 44-byte WAV header
const url = `wss://model-${process.env.MODEL_ID}.api.baseten.co/environments/production/websocket`;
const ws = new WebSocket(url, { headers: { Authorization: `Api-Key ${process.env.BASETEN_API_KEY}` } });

let printed = 0;
ws.on("open", () => {
  ws.send(JSON.stringify({ session_id: `demo-${Date.now()}`, max_speakers: Number(maxSpeakers) }));
  let off = 0;
  const tick = setInterval(() => {
    if (off >= pcm.length) {
      clearInterval(tick);
      ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
      return;
    }
    ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio: pcm.subarray(off, off + FRAME_BYTES).toString("base64") }));
    off += FRAME_BYTES;
  }, 100); // real-time pacing
});

ws.on("message", (data) => {
  const frame = JSON.parse(data);
  if (frame.type === "error") { console.error("server error:", frame.error); process.exit(1); }
  for (const seg of frame.segments.slice(printed)) {
    console.log(`[${seg.speaker} ${seg.start.toFixed(2)}-${seg.end.toFixed(2)}]${seg.overlap ? " (overlap)" : ""} ${seg.text}`);
  }
  printed = frame.segments.length;
  for (const p of frame.partial ?? []) console.log(`    … ${p.speaker}: ${p.text}`);
  if (frame.is_final) {
    console.log(`\nfinal: ${frame.num_speakers} speaker(s), ${frame.segments.length} turns`);
    ws.close();
  }
});
ws.on("error", (e) => { console.error(e); process.exit(1); });
