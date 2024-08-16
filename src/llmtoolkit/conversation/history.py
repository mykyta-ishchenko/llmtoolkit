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

    def __getitem__(self, key: int) -> ConversationMessage:
        """
        Retrieves a message by index.

        Args:
            key (int): The index of the message to retrieve.

        Returns:
            ConversationMessage: The message at the specified index.
        """
        return self.messages[key]

    def __setitem__(self, key: int, value: ConversationMessage) -> None:
        """
        Sets a message at a specific index.

        Args:
            key (int): The index where the message should be set.
            value (ConversationMessage): The message to set.
        """
        self.messages[key] = value

    def __iter__(self) -> Generator[ConversationMessage, None, None]:
        """
        Returns an iterator over the messages.

        Returns:
            Generator[ConversationMessage, None, None]: An iterator for the message list.
        """
        return iter(self.messages)

    def extend(self, values: list[ConversationMessage]) -> None:
        """
        Extends the message list with additional messages.

        Args:
            values (list[ConversationMessage]): The messages to add.
        """
        self.messages.extend(values)

    def append(self, value: ConversationMessage) -> None:
        """
        Appends a message to the end of the list.

        Args:
            value (ConversationMessage): The message to append.
        """
        self.messages.append(value)

    @classmethod
    def from_dict(cls, values: list[dict[str, str]]) -> "ConversationHistory":
        new_history = ConversationHistory()
        for value in values:
            new_history.append(ConversationMessage(**value))
        return new_history