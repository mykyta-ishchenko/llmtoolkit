from collections.abc import Generator

from llmtoolkit.core import Chain
from llmtoolkit.core.models import (
    ChainResponse,
    ConversationHistory,
)


class PromptChain(Chain):
    prompt: str

    def _prepare(self, conversation_history: ConversationHistory) -> None:
        conversation_history.add_assistant_message(content=self.prompt)
        return conversation_history

    def generate(self, conversation_history: ConversationHistory, **kwargs) -> ChainResponse:
        conversation_history = self._prepare(conversation_history)
        return self.chain.generate(conversation_history, **kwargs)

    def stream(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = self._prepare(conversation_history)
        return self.chain.stream(conversation_history, **kwargs)

    async def agenerate(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> ChainResponse:
        conversation_history = self._prepare(conversation_history)
        return await self.chain.agenerate(conversation_history, **kwargs)

    async def astream(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = self._prepare(conversation_history)
        return self.chain.astream(conversation_history, **kwargs)
