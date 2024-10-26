"""
Module for defining the base class for language models in the LLM-Toolkit library.

This module contains the `BaseLLMModel` class, which serves as an abstract base class for
all language models. It defines the common interface for generating text and retrieving
model information, with support for both synchronous and asynchronous operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel

from llmtoolkit.conversation.history import ConversationHistory


class Base(BaseModel):
    """
    A base configuration class providing shared functionality for model classes.

    Methods:
        get_model_info(): Returns basic information about the model's name and configuration.
    """

    model_name: str = "default"

    def get_model_info(self) -> dict[str, Any]:
        """
        Returns information about the model.

        Returns:
            dict[str, Any]: A dictionary containing the model's name and configuration.

        """
        return {"model_name": self.model_name}

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
        protected_namespaces = ()


class BaseLLMModel(ABC, Base):
    """
    Abstract base class for all language models in the LLM-Toolkit library.
    Defines a common interface for generating text and retrieving model information.
    """

    @abstractmethod
    def generate(self, prompt: str, conversation_history: ConversationHistory, **kwargs) -> str:
        """
        Abstract method for generating text based on a given prompt.

        Args:
            prompt (str): Input text prompt for generating a response.
            conversation_history (ConversationHistory): The conversation history.
            **kwargs (Any): Additional parameters for text generation.

        Returns:
            str: The generated text response.

        Notes:
            This method must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory, **kwargs
    ) -> Generator[str, None, None]:
        """
        Abstract method for generating text in a streaming manner.

        Args:
            prompt (str): Input text prompt for generating a response.
            conversation_history (ConversationHistory): The conversation history.
            **kwargs (Any): Additional parameters for text generation.

        Yields:
            str: The generated text response in streaming chunks.

        Notes:
            This method must be implemented by subclasses.
        """
        ...


class BaseAsyncLLMModel(ABC, Base):
    """
    Abstract base class for asynchronous language models in the LLM-Toolkit library.
    Defines a common interface for generating text and retrieving model information
    asynchronously.
    """

    @abstractmethod
    async def generate(
        self, prompt: str, conversation_history: ConversationHistory, **kwargs
    ) -> str:
        """
        Abstract asynchronous method for generating text based on a given prompt.

        Args:
            prompt (str): Input text prompt for generating a response.
            conversation_history (ConversationHistory): The conversation history.
            **kwargs (Any): Additional parameters for text generation.

        Returns:
            str: The generated text response.

        Notes:
            This method must be implemented by subclasses and is expected to be asynchronous.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory, **kwargs
    ) -> Generator[str, None, None]:
        """
        Abstract asynchronous method for generating text in a streaming manner.

        Args:
            prompt (str): Input text prompt for generating a response.
            conversation_history (ConversationHistory): The conversation history.
            **kwargs (Any): Additional parameters for text generation.

        Yields:
            str: The generated text response in streaming chunks.

        Notes:
            This method must be implemented by subclasses and is expected to be asynchronous.
        """
        ...
