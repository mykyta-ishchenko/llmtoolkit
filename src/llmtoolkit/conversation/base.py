"""
BaseConversation - abstract base class for managing language model conversations.

This module defines the `BaseConversation` and `BaseAsyncConversation` classes, which provide
an interface for generating responses and streaming outputs, managing conversation history,
and interacting with a language model in synchronous and asynchronous modes.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, Field

from .history import ConversationHistory


class BaseConversation(ABC, BaseModel):
    """
    Abstract base class for managing conversation history and interacting with a language model
    in a synchronous manner.

    Attributes:
        history (ConversationHistory): Manages the history of interactions in the conversation.
        llm (Any): A language model instance used for generating responses.
    """

    history: ConversationHistory = Field(default_factory=ConversationHistory)
    llm: Any

    @abstractmethod
    def chat(
        self,
        prompt: str,
    ) -> str:
        """
        Abstract method for generating a synchronous response based on a given prompt.

        Args:
            prompt (str): Input prompt for generating a response.

        Returns:
            str: The generated response from the language model.

        Notes:
            This method must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def stream(
        self,
        prompt: str,
    ) -> Generator[str, None, None]:
        """
        Abstract method for streaming a synchronous response based on a given prompt.

        Args:
            prompt (str): Input prompt for generating a streaming response.

        Yields:
            str: The generated response chunks from the language model in streaming format.

        Notes:
            This method must be implemented by subclasses.
        """
        ...


class BaseAsyncConversation(BaseModel):
    """
    Abstract base class for managing conversation history and interacting with a language model
    in an asynchronous manner.

    Attributes:
        history (ConversationHistory): Manages the history of interactions in the conversation.
        llm (Any): An asynchronous language model instance used for generating responses.
    """

    history: ConversationHistory = Field(default_factory=ConversationHistory)
    llm: Any

    @abstractmethod
    async def chat(
        self,
        prompt: str,
    ) -> str:
        """
        Abstract asynchronous method for generating a response based on a given prompt.

        Args:
            prompt (str): Input prompt for generating a response.

        Returns:
            str: The generated response from the language model.

        Notes:
            This method must be implemented by subclasses.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
    ) -> Generator[str, None, None]:
        """
        Abstract asynchronous method for streaming a response based on a given prompt.

        Args:
            prompt (str): Input prompt for generating a streaming response.

        Yields:
            str: The generated response chunks from the language model in streaming format.

        Notes:
            This method must be implemented by subclasses.
        """
        ...
