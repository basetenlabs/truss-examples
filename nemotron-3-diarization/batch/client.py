"""Diarize a recording with the Nemotron 3 Diarization batch endpoint.

    export BASETEN_API_KEY=...  MODEL_ID=...
    python client.py --url https://example.com/meeting.wav
    python client.py --file meeting.flac --latency low      # base64 upload

Prints one line per speaker turn: start, end, speaker.
"""

import argparse
import base64
import os
import sys

import requests


def diarize(audio: dict, latency: str = "offline") -> dict:
    url = f"https://model-{os.environ['MODEL_ID']}.api.baseten.co/environments/production/predict"
    resp = requests.post(
        url,
        headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
        json={"diarization_input": {"audio": audio, "latency": latency}},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="public or presigned URL of the audio file")
    src.add_argument(
        "--file", help="local audio file, sent as base64 (any ffmpeg-decodable format)"
    )
    ap.add_argument(
        "--latency", default="offline", choices=["offline", "low", "ultralow"]
    )
    args = ap.parse_args()

    if args.url:
        audio = {"url": args.url}
    else:
        with open(args.file, "rb") as f:
            audio = {"audio_b64": base64.b64encode(f.read()).decode()}

    result = diarize(audio, args.latency)
    for turn in result["turns"]:
        print(f"{turn['start']:8.2f}  {turn['end']:8.2f}  {turn['speaker']}")
    t = result["timing"]
    print(
        f"\n{result['num_speakers']} speaker(s), {t['audio_s']:.1f} s of audio, "
        f"profile={result['latency']}, server {t['total_ms']} ms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
