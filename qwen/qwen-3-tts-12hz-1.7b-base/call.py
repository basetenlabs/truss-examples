"""Client for Qwen3-TTS Base (voice cloning) via Truss predict endpoint.

Examples:
    # Single voice clone from URL reference
    python call.py --text "Hello, how are you?" \
        --ref-audio "https://example.com/reference.wav" \
        --ref-text "Transcript of the reference audio."

    # Voice clone from local file
    python call.py --text "Hello, how are you?" \
        --ref-audio ./reference.wav \
        --ref-text "Transcript of the reference audio."

    # x-vector only mode (faster, slightly less similar)
    python call.py --text "Hello, how are you?" \
        --ref-audio ./reference.wav \
        --ref-text "Transcript of the reference." \
        --x-vector-only

    # Batch mode (multiple texts at once)
    python call.py \
        --text "Hello world." "How are you doing today?" \
        --language Auto Auto \
        --ref-audio ./reference.wav \
        --ref-text "Transcript of the reference." \
        --output batch_out
"""

import argparse
import base64
import os
import subprocess
import tempfile

import httpx

DEFAULT_API_BASE = "https://model-w5ddzoj3.api.baseten.co/development/predict"
DEFAULT_API_KEY = os.getenv("BASETEN_API_KEY")


def convert_to_wav(audio_path: str) -> str:
    """Convert any audio file to 24kHz mono WAV via ffmpeg. Returns path to WAV."""
    if audio_path.lower().endswith(".wav"):
        return audio_path
    print(f"Converting {os.path.basename(audio_path)} -> WAV via ffmpeg...")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "24000", "-ac", "1", tmp.name],
        check=True,
        capture_output=True,
    )
    return tmp.name


def encode_audio_to_base64(audio_path: str) -> str:
    """Convert to WAV if needed, then encode as a base64 data URI."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    wav_path = convert_to_wav(audio_path)
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
    if wav_path != audio_path:
        os.unlink(wav_path)

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{audio_b64}"


def run_tts(args) -> None:
    """Run TTS voice cloning via the Truss predict endpoint."""
    is_batch = isinstance(args.text, list) and len(args.text) > 1

    ref_audio = args.ref_audio
    if ref_audio and not ref_audio.startswith(("http://", "https://", "data:")):
        ref_audio = encode_audio_to_base64(ref_audio)

    payload = {
        "text": args.text if is_batch else args.text[0],
        "language": args.language if is_batch else args.language[0],
        "ref_audio": ref_audio,
        "ref_text": args.ref_text,
        "x_vector_only_mode": args.x_vector_only,
        "max_new_tokens": args.max_new_tokens,
        "response_format": args.response_format,
    }

    for key in ("temperature", "top_k", "top_p", "repetition_penalty"):
        val = getattr(args, key, None)
        if val is not None:
            payload[key] = val

    print(f"Text: {payload['text']}")
    print(f"Language: {payload['language']}")
    print(f"x_vector_only: {args.x_vector_only}")
    print("Generating audio...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {args.api_key}",
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(args.api_base, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    if is_batch:
        data = response.json()
        if "error" in data:
            print(f"Error: {data['error']}")
            return

        os.makedirs(args.output, exist_ok=True)
        for i, b64_audio in enumerate(data["audio"]):
            audio_bytes = base64.b64decode(b64_audio)
            out_path = os.path.join(args.output, f"output_{i}.wav")
            with open(out_path, "wb") as f:
                f.write(audio_bytes)
            print(f"Saved: {out_path}")
    else:
        try:
            text = response.content.decode("utf-8")
            if text.startswith('{"error"'):
                print(f"Error: {text}")
                return
        except UnicodeDecodeError:
            pass

        output_path = args.output
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Audio saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Client for Qwen3-TTS Base (voice cloning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="Baseten API key (default: BASETEN_API_KEY env var)",
    )

    # Input
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        required=True,
        help="Text(s) to synthesize. Multiple values for batch mode.",
    )
    parser.add_argument(
        "--language",
        type=str,
        nargs="+",
        default=["Auto"],
        help="Language(s): Auto, Chinese, English, etc. (default: Auto)",
    )

    # Voice clone parameters
    parser.add_argument(
        "--ref-audio",
        type=str,
        required=True,
        help="Reference audio: local file path or URL",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference transcript: inline text or path to a .txt file (required unless --x-vector-only)",
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use x-vector only mode (no ICL, faster but less similar)",
    )

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)

    # Output
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        choices=["wav", "flac"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="tts_output.wav",
        help="Output file path (single) or directory (batch). Default: tts_output.wav",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.x_vector_only and not args.ref_text:
        print("Error: --ref-text is required unless --x-vector-only is set")
        exit(1)

    if args.ref_text and os.path.isfile(args.ref_text):
        with open(args.ref_text, "r") as f:
            args.ref_text = f.read().strip()

    if len(args.language) == 1 and len(args.text) > 1:
        args.language = args.language * len(args.text)

    run_tts(args)
