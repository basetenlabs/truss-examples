"""OpenAI-compatible client for Qwen3-TTS via /v1/audio/speech endpoint.

This script demonstrates how to use the OpenAI-compatible speech API
to generate audio from text using Qwen3-TTS models.

The API URL is automatically selected based on the task type:
    - CustomVoice: Uses predefined speaker voices
    - VoiceDesign: Creates voice from description
    - Base: Voice cloning from reference audio

Examples:
    # CustomVoice task (default - predefined speaker)
    python call.py --text "Hello, how are you?" --voice Vivian

    # CustomVoice with emotion instruction
    python call.py --text "I'm so happy!" --voice Vivian \
        --instructions "Speak with excitement"

    # VoiceDesign task (voice from description)
    python call.py --text "Hello world" \
        --task-type VoiceDesign \
        --instructions "A warm, friendly female voice"

    # Base task (voice cloning)
    python call.py --text "Hello world" \
        --task-type Base \
        --ref-audio "https://example.com/reference.wav" \
        --ref-text "This is the reference transcript"

    # Override API URL manually
    python call.py --text "Hello" --api-base "https://custom-url.com"
"""

import argparse
import base64
import os

import httpx

DEFAULT_API_URL_CUSTOM_VOICE = "<CUSTOM_VOICE_API_URL>"  # Custom Voice https://model-....api.baseten.co/deployment/.../sync/
DEFAULT_API_URL_VOICE_DESIGN = "<VOICE_DESIGN_API_URL>"  # Voice Design https://model-....api.baseten.co/deployment/.../sync/
DEFAULT_API_URL_BASE = "<BASE_API_URL>"  # Voice Clone https://model-....api.baseten.co/deployment/.../sync/
DEFAULT_API_KEY = os.environ.get("BASETEN_API_KEY", "")

def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect MIME type from extension
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    else:
        mime_type = "audio/wav"  # Default

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def get_api_url_for_task(task_type: str) -> str:
    """Get the appropriate API URL for the given task type."""
    url_map = {
        "CustomVoice": DEFAULT_API_URL_CUSTOM_VOICE,
        "VoiceDesign": DEFAULT_API_URL_VOICE_DESIGN,
        "Base": DEFAULT_API_URL_BASE,
    }
    return url_map.get(task_type, DEFAULT_API_URL_CUSTOM_VOICE)


def get_model_for_task(task_type: str) -> str:
    """Get the appropriate model path for the given task type."""
    model_map = {
        "CustomVoice": "/app/model_cache/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "VoiceDesign": "/app/model_cache/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Base": "/app/model_cache/Qwen3-TTS-12Hz-1.7B-Base",
    }
    return model_map.get(task_type, model_map["CustomVoice"])


def run_tts_generation(args) -> None:
    """Run TTS generation via OpenAI-compatible /v1/audio/speech API."""

    # Determine API URL based on task type (or use explicit override)
    if args.api_base:
        api_base = args.api_base
    else:
        api_base = get_api_url_for_task(args.task_type)

    # Determine model based on task type (or use explicit override)
    model = args.model if args.model else get_model_for_task(args.task_type)

    # Build request payload
    payload = {
        "model": model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
    }

    # Add optional parameters
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.task_type:
        payload["task_type"] = args.task_type
    if args.language:
        payload["language"] = args.language
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens



    # Voice clone parameters (Base task)
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            import base64

            # Download the reference audio
            ref_audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
            response = httpx.get(ref_audio_url)
            audio_b64 = base64.b64encode(response.content).decode("utf-8")
            ref_audio_data_url = f"data:audio/wav;base64,{audio_b64}"

            # Use this in your call instead of the URL
            payload["ref_audio"] = ref_audio_data_url
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text

    # Auto-enable x-vector only mode in Base task if no ref_text provided
    use_x_vector = args.x_vector_only or (args.task_type == "Base" and not args.ref_text)
    if use_x_vector:
        payload["x_vector_only_mode"] = True

    print(f"Task type: {args.task_type}")
    print(f"API URL: {api_base}")
    print(f"Model: {model}")
    if args.task_type == "Base":
        print(f"X-vector only mode: {use_x_vector}")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    print("Generating audio...")

    # Make the API call
    api_url = f"{api_base.rstrip('/')}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {args.api_key}",
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Save audio response
    output_path = args.output or "tts_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible client for Qwen3-TTS via /v1/audio/speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Task configuration (first, since URL depends on it)
    parser.add_argument(
        "--task-type",
        "-t",
        type=str,
        default="CustomVoice",
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="TTS task type (default: CustomVoice)",
    )

    # Server configuration
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (overrides task-based URL selection)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key (default: BASETEN_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model name/path (auto-selected based on task type if not specified)",
    )

    # Input text
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )

    # Voice/speaker
    parser.add_argument(
        "--voice",
        type=str,
        default="alloy",
        help="Speaker/voice name (default: Vivian). Options: Vivian, Ryan, etc.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language: Auto, Chinese, English, etc.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Voice style/emotion instructions",
    )

    # Base (voice clone) parameters
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio file path or URL for voice cloning (Base task)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference audio transcript for voice cloning (Base task)",
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use x-vector only mode for voice cloning (no ICL)",
    )

    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate",
    )

    # Output
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (default: tts_output.wav)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tts_generation(args)