"""
BaseConversation - abstract base class for managing language model conversations.
Defines the `BaseConversation` class with methods for generating responses and streaming.
"""

from abc import abstractmethod
from collections.abc import Generator

from pydantic import BaseModel, Field

from llmtoolkit.llms import BaseLLMModel

from .history import ConversationHistory


class BaseConversation(BaseModel):
    """Abstract base class for managing conversation history
    and interacting with a language model."""

    history: ConversationHistory = Field(default_factory=ConversationHistory)
    llm: BaseLLMModel

    @abstractmethod
    async def chat(
        self,
        prompt: str,
    ) -> str:
        """
        Abstract method for generating a response.

        Args:
            prompt (str): The prompt for the model.

        Returns:
            str: The model's response.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
    ) -> Generator[str, None, None]:
        """
        Abstract method for streaming responses.

        Args:
            prompt (str): The prompt for the model.

        Returns:
            Generator[str, None, None]: A generator of the model's responses.
        """
        ...
