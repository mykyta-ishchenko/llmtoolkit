"""
Module for managing conversation messages and history in the LLM-Toolkit library.

This module provides the `Roles` enum for defining participant roles,
and the `ConversationMessage` and `ConversationHistory` classes for managing
individual messages and conversation history.
"""

from collections.abc import Generator
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Roles(Enum):
    """
    Enum for defining roles in a conversation.

    Members:
        USER: Represents the end-user in a conversation.
        ASSISTANT: Represents the assistant or model responding to the user.
        SYSTEM: Represents a system message, typically used for setting context or instructions.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """
    Represents a single message in a conversation.

    Attributes:
        role (str): The role of the sender in the conversation (e.g., user, assistant, or system).
        content (str): The main content of the message.
        context (Optional[str]): Additional context for the message, default is None.
    """

    role: str
    content: str
    context: str | None = None


class ConversationHistory(BaseModel):
    """
    Manages a list of conversation messages, providing methods to interact with the history.

    Attributes:
        messages (list[ConversationMessage]): List of messages in the conversation history.
    """

    messages: list[ConversationMessage] = Field(default_factory=list)

    def __getitem__(self, key: int) -> ConversationMessage:
        """
        Retrieves a message at the specified index.

        Args:
            key (int): Index of the message to retrieve.

        Returns:
            ConversationMessage: The message at the specified index.
        """
        return self.messages[key]

    def __setitem__(self, key: int, value: ConversationMessage) -> None:
        """
        Sets or replaces a message at a specified index.

        Args:
            key (int): Index where the message should be set.
            value (ConversationMessage): The message to set at the specified index.
        """
        self.messages[key] = value

    def __iter__(self) -> Generator[ConversationMessage, None, None]:
        """
        Provides an iterator over the conversation messages.

        Yields:
            ConversationMessage: Each message in the conversation history.
        """
        return iter(self.messages)

    def extend(self, values: list[ConversationMessage]) -> None:
        """
        Extends the conversation history with a list of additional messages.

        Args:
            values (list[ConversationMessage]): Messages to add to the history.
        """
        self.messages.extend(values)

    def append(self, value: ConversationMessage) -> None:
        """
        Appends a single message to the end of the conversation history.

        Args:
            value (ConversationMessage): The message to append to the history.
        """
        self.messages.append(value)

    def __len__(self) -> int:
        """
        Returns the number of messages in the conversation history.

        Returns:
            int: The length of the conversation history.
        """
        return len(self.messages)

    def to_dict(self) -> list[dict[str, Any]]:
        """
        Converts the conversation history to a list of dictionaries.

        Returns:
            list[dict[str, Any]]: List of messages in dictionary form.
        """
        return self.model_dump().get("messages")
