import asyncio
import logging
import time
from pathlib import Path

import asr_workers.whisper
import truss_chains as chains
import utilities.data_types as data_types
import utilities.helpers as helpers
import utilities.utilities as utilities

# Configure the logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@chains.mark_entrypoint
class TranscribeWithVad(chains.ChainletBase):
    """Transcribes one file end-to-end and sends results to webhook."""

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            apt_requirements=["ffmpeg"],
            # TODO: pin the version of the package
            pip_requirements=[
                "pandas",
                "google-auth",
                "google-cloud-bigquery",
                "torch",
                "torchaudio",
                "toml",
                "soundfile==0.10.2",
            ],
        ),
        compute=chains.Compute(predict_concurrency=128),
    )
    _context: chains.DeploymentContext

    _whisper: asr_workers.whisper.WhisperModel

    def __init__(
        self,
        whisper: asr_workers.whisper.WhisperModel = chains.depends(
            asr_workers.whisper.WhisperModel, retries=2
        ),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        from chunker.vad import VadOptions

        self._context = context
        self._whisper = whisper
        # TODO: expose these parameters as input
        self._vad_options = VadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            max_speech_duration_s=30,
            min_silence_duration_ms=3000,
            speech_pad_ms=500,
        )

    async def run_remote(
        self,
        whisper_input: data_types.InputSchema,
    ) -> data_types.WhisperResult:
        from chunker.vad import wav_chunker_vad

        tasks = []
        seg_infos = []

        async with helpers.AudioStreamer(whisper_input.audio) as audio_streamer:
            start_time = time.time()
            with utilities.measure("Vad chunk time"):
                chunk_stream = wav_chunker_vad(audio_streamer, self._vad_options)
            async for seg_info, audio_b64 in chunk_stream:
                print(f"Time to first chunk: {time.time() - start_time}")

                async def transcribe(audio_b64):
                    # Step 1: Run Whisper transcription
                    whisper_result = await self._whisper.run_remote(
                        data_types.WhisperInput(
                            audio_b64=audio_b64,
                            **whisper_input.whisper_params.model_dump(),
                        )
                    )
                    return whisper_result

                tasks.append(asyncio.ensure_future(transcribe(audio_b64)))
                seg_infos.append(seg_info)

        results = await utilities.gather(tasks)

        all_segments = []
        language_code = None

        for i, whisper_results in enumerate(results):
            if len(whisper_results.segments) > 0:
                for _, segment in enumerate(whisper_results.segments):
                    segment.start_time_sec += seg_infos[i].start_time_sec or 0
                    segment.end_time_sec += seg_infos[i].start_time_sec or 0

            language_code = whisper_results.language_code
            all_segments.extend(whisper_results.segments)

        return data_types.WhisperResult(
            segments=all_segments,
            language_code=language_code,
        )
