import math
import subprocess
import tempfile
from collections.abc import AsyncGenerator, Generator
from typing import Any

from openai import NOT_GIVEN, AsyncOpenAI, OpenAI
from pydantic import PrivateAttr

from llmtoolkit.core import UNSET
from llmtoolkit.core.models import ASRResponse
from llmtoolkit.exc import FfmpegError, NotImplementedToolkitError

from .base_whisper import BaseWhisper


class OpenAIWhisper(BaseWhisper):
    api_key: str = "-"
    host: str | None = None

    _max_file_size = 25_000_000
    _overlap_seconds = 3
    _silence_threshold = -40

    _client: OpenAI = PrivateAttr()
    _async_client: AsyncOpenAI = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.host)
        self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.host)

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        return float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ]
            ).strip()
        )

    @staticmethod
    def _get_bit_rate(audio_path: str) -> float:
        return float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=bit_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ]
            ).strip()
        )

    def _split_audio(self, audio_path: str) -> list[str]:
        bit_rate = self._get_bit_rate(audio_path)
        duration = self._get_audio_duration(audio_path)

        chunk_duration_s = (self._max_file_size * 8.0) / bit_rate * 0.9
        num_chunks = math.ceil(duration / (chunk_duration_s - self._overlap_seconds))

        chunks = []
        start_time = 0

        for i in range(num_chunks):
            end_time = min(start_time + chunk_duration_s, duration)
            chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            subprocess.call(
                [
                    "ffmpeg",
                    "-i",
                    audio_path,
                    "-ss",
                    str(start_time),
                    "-to",
                    str(end_time),
                    "-y",
                    chunk_file.name,
                ]
            )
            chunks.append(chunk_file.name)
            start_time = end_time - self._overlap_seconds

        return chunks

    def _prepare_audio_chunks(self, audio: str | bytes) -> tuple[list[str], list[str]]:
        try:
            temp_audio_path = self._save_to_temp_file(audio)
            chunks = self._split_audio(temp_audio_path)
            temp_files = [temp_audio_path] + chunks
            return chunks, temp_files
        except FileNotFoundError as e:
            if self._ffmpeg_installation_error in str(e):
                raise FfmpegError
            raise e

    def transcribe(self, audio: str | bytes, language: str = UNSET) -> ASRResponse:
        chunks, temp_files = self._prepare_audio_chunks(audio)

        full_transcription = ""
        try:
            for chunk_path in chunks:
                with open(chunk_path, "rb") as file:
                    transcription = self._client.audio.transcriptions.create(
                        model=self.model_name,
                        file=file,
                        language=NOT_GIVEN if language is UNSET else language,
                    )
                    full_transcription += transcription.text.strip() + " "
        finally:
            self._cleanup_files(temp_files)

        return ASRResponse(text=full_transcription.strip())

    async def async_transcribe(self, audio: str | bytes, language: str = UNSET) -> ASRResponse:
        chunks, temp_files = self._prepare_audio_chunks(audio)

        full_transcription = ""
        try:
            for chunk_path in chunks:
                with open(chunk_path, "rb") as file:
                    transcription = await self._async_client.audio.transcriptions.create(
                        model=self.model_name,
                        file=file,
                        language=NOT_GIVEN if language is UNSET else language,
                    )
                    full_transcription += transcription.text.strip() + " "
        finally:
            self._cleanup_files(temp_files)

        return ASRResponse(text=full_transcription.strip())

    def stream(
        self, audio: str | bytes, language: str = UNSET
    ) -> Generator[ASRResponse, None, None]:
        raise NotImplementedToolkitError

    async def async_stream(
        self, audio: str | bytes, language: str = UNSET
    ) -> AsyncGenerator[ASRResponse, None]:
        raise NotImplementedToolkitError
