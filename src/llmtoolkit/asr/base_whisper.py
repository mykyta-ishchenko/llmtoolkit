import os
import subprocess
import tempfile
from abc import ABC

from llmtoolkit.core import ASRModel
from llmtoolkit.exc import UnsupportedFormatError


class BaseWhisper(ASRModel, ABC):
    _ffmpeg_installation_error = "[Errno 2] No such file or directory: 'ffmpeg'"

    @staticmethod
    def _save_to_temp_file(audio: str | bytes) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        if isinstance(audio, str):
            subprocess.call(["ffmpeg", "-i", audio, "-y", temp_file.name])
        elif isinstance(audio, bytes):
            with open(temp_file.name, "wb") as f:
                f.write(audio)
        else:
            raise UnsupportedFormatError
        return temp_file.name

    @staticmethod
    def _cleanup_files(files: list[str]):
        for file in files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
