"""
Module for managing conversation messages and history in the LLM-Toolkit library.
"""

from collections.abc import Generator

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """
    Represents a message in a conversation.

    Attributes:
        role (str): The role of the sender (e.g., user, assistant).
        content (str): The content of the message.
        context (str): Optional context for the message, default is an empty string.
    """

    role: str
    content: str
    context: str = ""


class ConversationHistory(BaseModel):
    """
    Manages a list of conversation messages.

    Attributes:
        messages (list[ConversationMessage]): The list of messages in the conversation.
    """

    messages: list[ConversationMessage] = Field(default_factory=list)

    def __len__(self):
        """Length of the history."""
        return len(self.messages)
