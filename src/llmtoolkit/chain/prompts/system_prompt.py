from llmtoolkit.core.models import (
    ConversationHistory,
    ConversationMessage,
    Roles,
)

from .prompt import PromptChain


class SystemPromptChain(PromptChain):
    prompt: str

    def _prepare(self, conversation_history: ConversationHistory) -> ConversationHistory:
        conversation_history = conversation_history or ConversationHistory()
        conversation_history.insert(0, ConversationMessage(role=Roles.SYSTEM, content=self.prompt))
        return conversation_history
