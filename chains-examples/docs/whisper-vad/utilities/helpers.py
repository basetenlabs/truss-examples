import asyncio
import base64
import logging
import multiprocessing
import os
import signal
import tempfile
from asyncio import subprocess
from typing import Optional

import utilities.data_types as data_types

# Configure the logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


WAV_SAMPLING_RATE_HZ = 16_000  # Frequency that Whisper is trained on.


class AudioStreamer:
    """Contextmanager reading subprocess streams asynchronously and
    properly handling errors.

    Given a media URL and time boundaries, emits on the `wav_stream`-attribute
    an async byte stream of the extracted and resampled mono-channel waveform.
    """

    _audio_src: data_types.AudioSource
    _process: Optional[subprocess.Process]
    _temp_file: Optional[tempfile.TemporaryFile]

    def __init__(
        self,
        audio_src: data_types.AudioSource,
    ) -> None:
        if (audio_src.audio_b64 is not None) == (audio_src.url is not None):
            raise ValueError("One of audio_b64 or url must be provided, but not both")
        self._audio_src = audio_src
        # TODO: handle different start times for byte ranges
        self.start_time = 0
        self.duration_sec = None
        self._process = None
        self._temp_file = None

    async def __aenter__(self) -> "AudioStreamer":
        if self._audio_src.audio_b64 is not None:
            audio = base64.b64decode(self._audio_src.audio_b64.encode("utf-8"))
            self._temp_file = tempfile.NamedTemporaryFile(delete=False)
            self._temp_file.write(audio)
            path_or_url = self._temp_file.name
        else:
            path_or_url = self._audio_src.url

        num_threads = (multiprocessing.cpu_count() - 1) * 2
        ffmpeg_command = "ffmpeg " + (
            f'-i "{path_or_url}" '
            f"-threads {num_threads} "
            f"-vn "  # Disables video recording.
            f"-acodec pcm_s16le "  # Audio codec: PCM signed 16-bit little endian.
            f"-ar {WAV_SAMPLING_RATE_HZ} "  # -ar: Sets the audio sample rate.
            f"-ac 1 "  # Average channels to mono.
            f"-af 'pan=mono|c0=0.5*c0+0.5*c1' "  # -af "pan=mono|c0=0.5*c0+0.5*c1": Applies an audio filter to downmix to mono
            #  by averaging the channels. The pan filter is used here to control the mix.
            #  c0=0.5*c0+0.5*c1 takes the first channel (c0) and averages it with the second
            #  channel (c1). For sources with more than two channels, adjust the formula to
            #  include them accordingly.
            f"-f wav "  # -f wav: Specifies the output format to be WAV.
            f"-"  # -: write output to stdout.
        )
        self._process = await subprocess.create_subprocess_shell(
            ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert self._process.stdout is not None
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self._process is not None
        if self._temp_file is not None:
            self._temp_file.close()
            os.unlink(self._temp_file.name)

        logging.debug(f"Exiting context, exc_type={exc_type}, exc={exc}")
        if self._process.returncode is None:
            logging.debug("Sending sigterm to sub-process.")
            self._process.terminate()

        try:
            logging.debug("Waiting for sub-process.")
            await asyncio.wait_for(self._process.wait(), 5.0)
        except asyncio.TimeoutError as e:
            logging.debug("Waiting timed out, killing sub-process.")
            self._process.kill()
            await self._process.wait()  # Wait again after killing
            stderr = (
                (await self._process.stderr.read()).decode()
                if self._process.stderr
                else "No stderr available."
            )
            raise ChildProcessError(
                "FFMPEG hangs after terminating. Stderr:\n" f"{stderr}"
            ) from e

        logging.debug(f"return code={self._process.returncode}.")
        # The sub-proces did not crash (it succeeded, or we terminated it) - but
        # there was an exception from the within the context block.
        if exc and self._process.returncode in (
            0,
            -signal.SIGTERM,
            -signal.SIGKILL,
        ):
            logging.debug("Re-raising exception from main process.")
            raise exc

        # The subprocess crashed.
        if self._process.returncode != 0:
            logging.debug("Raising exception from failed sub-process.")
            stderr = (
                (await self._process.stderr.read()).decode()
                if self._process.stderr
                else "No stderr available."
            )
            raise ChildProcessError(
                "FFMPEG error during video download and "
                f"wav extraction. Stderr:\n{stderr}"
            ) from exc  # In case there was also an error in the context-block, chain.

        # E.g. there was an exception AND we couldn't terminate the sub-process.
        elif exc:
            logging.debug(f"Handling of exception not otherwise covered: {exc}")
            raise exc

    @property
    def wav_stream(self) -> asyncio.StreamReader:
        assert self._process is not None
        stdout = self._process.stdout
        assert stdout is not None
        return stdout
