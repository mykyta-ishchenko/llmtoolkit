from abc import ABC, abstractmethod
from collections.abc import Generator

from pydantic import BaseModel

from .models import ASRResponse
from .objects import UNSET


class ASRModel(ABC, BaseModel):
    model_name: str = "default"

    @abstractmethod
    def transcribe(
        self, audio: str | bytes, filetype: str, language: str = UNSET
    ) -> ASRResponse: ...

    @abstractmethod
    async def atranscribe(
        self, audio: str | bytes, filetype: str, language: str = UNSET
    ) -> ASRResponse: ...

    @abstractmethod
    def stream(
        self, audio: str | bytes, filetype: str, language: str = UNSET
    ) -> Generator[ASRResponse, None, None]: ...

    @abstractmethod
    async def astream(
        self, audio: str | bytes, filetype: str, language: str = UNSET
    ) -> Generator[ASRResponse, None, None]: ...

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
        protected_namespaces = ()
