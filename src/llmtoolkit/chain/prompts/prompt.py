from collections.abc import Generator

from llmtoolkit.chain.base import Chain
from llmtoolkit.core.models import (
    ChainResponse,
    ConversationHistory,
    ConversationMessage,
    Roles,
)


class PromptChain(Chain):
    prompt: str

    def _prepare(self, conversation_history: ConversationHistory | None) -> ConversationHistory:
        conversation_history = conversation_history or ConversationHistory()
        conversation_history.append(ConversationMessage(role=Roles.ASSISTANT, content=self.prompt))
        return conversation_history

    def generate(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> ChainResponse:
        conversation_history = self._prepare(conversation_history)
        return self.chain.generate(conversation_history, **kwargs)

    def generate_stream(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = self._prepare(conversation_history)
        return self.chain.generate_stream(conversation_history, **kwargs)

    async def async_generate(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> ChainResponse:
        conversation_history = self._prepare(conversation_history)
        return await self.chain.async_generate(conversation_history, **kwargs)

    async def async_generate_stream(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = self._prepare(conversation_history)
        return self.chain.async_generate_stream(conversation_history, **kwargs)
