from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Optional

from pydantic import BaseModel

from .models import ChainResponse, ConversationHistory


class Chain(ABC, BaseModel):
    chain: Optional["Chain"] = None

    @abstractmethod
    def generate(self, conversation_history: ConversationHistory, **kwargs) -> ChainResponse: ...

    @abstractmethod
    async def async_generate(
        self, conversation_history: ConversationHistory, **kwargs
    ) -> ChainResponse: ...

    @abstractmethod
    def generate_stream(
        self, conversation_history: ConversationHistory, **kwargs
    ) -> Generator[ChainResponse, None, None]: ...

    @abstractmethod
    async def async_generate_stream(
        self, conversation_history: ConversationHistory, **kwargs
    ) -> Generator[ChainResponse, None, None]: ...

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
        protected_namespaces = ()
