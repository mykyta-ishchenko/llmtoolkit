"""
Module for managing conversation messages and history in the LLM-Toolkit library.
"""

from collections.abc import Generator

from pydantic import BaseModel


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


class ConversationHistory:
    """
    Manages a list of conversation messages.

    Attributes:
        messages (list[ConversationMessage]): The list of messages in the conversation.
    """

    def __init__(self, messages: list[ConversationMessage] = None):
        """
        Initializes the conversation history with an optional list of messages.

        Args:
            messages (list[ConversationMessage], optional): Initial list of messages.
                Defaults to an empty list.
        """
        self._messages = messages or []

    def __getitem__(self, key: int) -> ConversationMessage:
        """
        Retrieves a message by index.

        Args:
            key (int): The index of the message to retrieve.

        Returns:
            ConversationMessage: The message at the specified index.
        """
        return self._messages[key]

    def __setitem__(self, key: int, value: ConversationMessage) -> None:
        """
        Sets a message at a specific index.

        Args:
            key (int): The index where the message should be set.
            value (ConversationMessage): The message to set.
        """
        self._messages[key] = value

    def __iter__(self) -> Generator[ConversationMessage, None, None]:
        """
        Returns an iterator over the messages.

        Returns:
            Generator[ConversationMessage, None, None]: An iterator for the message list.
        """
        return iter(self._messages)

    def extend(self, values: list[ConversationMessage]) -> None:
        """
        Extends the message list with additional messages.

        Args:
            values (list[ConversationMessage]): The messages to add.
        """
        self._messages.extend(values)

    def append(self, value: ConversationMessage) -> None:
        """
        Appends a message to the end of the list.

        Args:
            value (ConversationMessage): The message to append.
        """
        self._messages.append(value)

    @classmethod
    def from_dict(cls, values: list[dict[str, str]]) -> "ConversationHistory":
        new_history = ConversationHistory()
        for value in values:
            new_history.append(ConversationMessage(**value))

    def model_dump(self):
        """
        Returns a dictionary representation of the message list.

        Returns:
            dict: A dictionary with a single key 'messages' mapping to a
                list of message representations.
        """
        return {"messages": [message.model_dump() for message in self._messages]}

    def __repr__(self):
        """
        Returns a string representation of the conversation history.

        Returns:
            str: A string representation of the messages in the history.
        """
        return f"messages=({', '.join([str(message) for message in self._messages])})"
