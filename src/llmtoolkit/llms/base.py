"""
Module for defining the base class for language models in the LLM-Toolkit library.

This module contains the `AbstractLLMModel` class, which serves as an abstract base class for
all large language models. It defines the common interface for generating text and retrieving
model information, with support for asynchronous operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel

from llmtoolkit.conversation.history import ConversationHistory


class Base(BaseModel):
    def get_model_info(self) -> dict[str, Any]:
        """
        Returns information about the model.

        Returns:
            Dict[str, Any]: A dictionary containing the model's name and configuration.
        """
        return {"model_name": self.model_name}

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
        protected_namespaces = ()


class BaseLLMModel(ABC, Base):
    """
    Abstract base class for all language models in the LLM-Toolkit library.
    Defines the common interface for generating text and retrieving model information.

    Attributes:
        model_name (str): The name of the model.
    """

    model_name: str

    @abstractmethod
    def generate(self, prompt: str, conversation_history: ConversationHistory, **kwargs) -> str:
        """
        Abstract method for generating text based on a given prompt.

        Args:
            prompt (str): The input text prompt for the model.
            conversation_history (ConversationHistory): The conversation history.
            **kwargs (Any): Additional parameters for the text generation.

        Returns:
            str: The generated text.

        Notes:
            This method must be implemented by any subclass.
        """
        ...

    @abstractmethod
    def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory, **kwargs
    ) -> Generator[str, None, None]:
        """
        Abstract method for generating text in a stream.

        Args:
            prompt (str): The input text prompt for the model.
            conversation_history (ConversationHistory): The conversation history.
            **kwargs (Any): Additional parameters for the text generation.

        Returns:
            Generator[str, None, None]: Generator with answer from model.

        Notes:
            This method must be implemented by any subclass.
        """
        ...


class BaseAsyncLLMModel(ABC, Base):
    """
    Abstract base class for all language models in the LLM-Toolkit library.
    Defines the common interface for generating text and retrieving model information.
    """

    model_name: str
    """The name of the model"""

    @abstractmethod
    async def generate(
        self, prompt: str, conversation_history: ConversationHistory, **kwargs
    ) -> str:
        """
        Abstract method for generating text based on a given prompt asynchronously.

        Args:
            prompt (str): The input text prompt for the model.
            conversation_history (ConversationHistory): The conversation history.
            **kwargs (Any): Additional parameters for the text generation.

        Returns:
            str: The generated text.

        Notes:
            This method must be implemented by any subclass and should be an asynchronous operation.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory, **kwargs
    ) -> Generator[str, None, None]:
        """
        Abstract method for generating text in a stream asynchronously.

        Args:
            prompt (str): The input text prompt for the model.
            conversation_history (ConversationHistory): The conversation history.
            **kwargs (Any): Additional parameters for the text generation.

        Returns:
            Generator[str, None, None]: Generator with answer from model.

        Notes:
            This method must be implemented by any subclass and should be an asynchronous operation.
        """
        ...
