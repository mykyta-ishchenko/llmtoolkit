from collections.abc import AsyncGenerator, Generator
from typing import Any

import whisper
from pydantic import PrivateAttr

from llmtoolkit.core import UNSET
from llmtoolkit.core.models import ASRResponse
from llmtoolkit.exc import NotImplementedToolkitError

from .base_whisper import BaseWhisper


class LocalWhisper(BaseWhisper):
    _model = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._model = whisper.load_model(self.model_name)

    def transcribe(self, audio: str | bytes, filetype: str, language: str = UNSET) -> ASRResponse:
        temp_audio_path = self._save_to_temp_file(audio, filetype)
        try:
            result = self._model.transcribe(
                temp_audio_path, language=None if language is UNSET else language
            )
            transcription = result["text"].strip()
        finally:
            self._cleanup_files([temp_audio_path])

        return ASRResponse(text=transcription)

    async def async_transcribe(
        self, audio: str | bytes, filetype: str, language: str = UNSET
    ) -> ASRResponse:
        return self.transcribe(audio, filetype, language)

    def stream(
        self, audio: str | bytes, filetype: str, language: str = UNSET
    ) -> Generator[ASRResponse, None, None]:
        raise NotImplementedToolkitError

    async def async_stream(
        self, audio: str | bytes, filetype: str, language: str = UNSET
    ) -> AsyncGenerator[ASRResponse, None]:
        raise NotImplementedToolkitError
