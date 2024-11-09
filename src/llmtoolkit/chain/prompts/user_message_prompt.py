from llmtoolkit.core.models import (
    ConversationHistory,
    ConversationMessage,
    Roles,
)

from .prompt import PromptChain


class UserMessagePromptChain(PromptChain):
    prompt: str

    def _prepare(self, conversation_history: ConversationHistory) -> ConversationHistory:
        if conversation_history:
            content = conversation_history[-1].content
            conversation_history[-1].content = self.prompt.format(content)
        else:
            conversation_history = ConversationHistory(
                messages=[ConversationMessage(role=Roles.USER, content=self.prompt)]
            )
        return conversation_history
