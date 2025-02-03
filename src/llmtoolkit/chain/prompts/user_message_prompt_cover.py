from llmtoolkit.core.enums import Roles
from llmtoolkit.core.models import (
    ConversationHistory,
)

from .prompt import PromptChain


class UserMessagePromptCoverChain(PromptChain):
    prompt: str

    def _prepare(self, conversation_history: ConversationHistory) -> ConversationHistory:
        for i in range(len(conversation_history) - 1, -1, -1):
            if conversation_history[i].role != Roles.USER:
                continue
            content = conversation_history[i].content
            conversation_history[i].content = self.prompt.format(content)
        return conversation_history
