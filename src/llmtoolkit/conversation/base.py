from abc import ABC, abstractmethod
from collections.abc import Generator

from pydantic import BaseModel, Field

from llmtoolkit.chain import Chain
from llmtoolkit.core.models import ChainResponse, ConversationHistory


class BaseConversation(ABC, BaseModel):
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    chain: Chain

    @abstractmethod
    def chat(self, prompt: str, **kwargs) -> ChainResponse: ...

    @abstractmethod
    async def async_chat(self, prompt: str, **kwargs) -> ChainResponse: ...

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[ChainResponse, None, None]: ...

    @abstractmethod
    async def async_stream(self, prompt: str, **kwargs) -> Generator[ChainResponse, None, None]: ...
