from abc import ABC, abstractmethod
from collections.abc import Generator

from pydantic import BaseModel, Field

from llmtoolkit.core import Chain
from llmtoolkit.core.models import ChainResponse, ConversationHistory


class BaseConversation(ABC, BaseModel):
    chain: Chain
    history: ConversationHistory = Field(default_factory=ConversationHistory)

    @abstractmethod
    def chat(
        self,
        prompt: str,
        **kwargs,
    ) -> ChainResponse: ...

    @abstractmethod
    async def achat(
        self,
        prompt: str,
        **kwargs,
    ) -> ChainResponse: ...

    @abstractmethod
    def stream(
        self,
        prompt: str,
        **kwargs,
    ) -> Generator[ChainResponse, None, None]: ...

    @abstractmethod
    async def astream(
        self,
        prompt: str,
        **kwargs,
    ) -> Generator[ChainResponse, None, None]: ...
