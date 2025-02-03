from llmtoolkit.core.models import (
    ConversationHistory,
)

from .prompt import PromptChain


class SystemPromptChain(PromptChain):
    prompt: str

    def _prepare(self, conversation_history: ConversationHistory) -> ConversationHistory:
        conversation_history.set_system_message(content=self.prompt)
        return conversation_history
