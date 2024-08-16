"""
Module for defining the base class for language models in the LLM-Toolkit library.

This module contains the `BaseLLMModel` class, which serves as an abstract base class for
all language models. It defines the common interface for generating text and retrieving
model information, with support for asynchronous operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from llmtoolkit.conversation import ConversationHistory


class BaseLLMModel(ABC):
    """
    Abstract base class for all language models in the LLM-Toolkit library.
    Defines the common interface for generating text and retrieving model information.

    Attributes:
        model_name (str): The name of the model.
        config (dict): Configuration parameters for the model.
    """

    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initializes the BaseLLMModel instance with configuration parameters.

        Args:
            model_name (str): The name of the model.
            **kwargs: Configuration parameters for the model.
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    async def generate(
        self, prompt: str, conversation_history: ConversationHistory
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
        self, prompt: str, conversation_history: ConversationHistory
    ):
        """
        Abstract method for generating text in a stream asynchronously.

        Args:
            prompt (str): The input text prompt for the model.
            conversation_history (ConversationHistory): The conversation history.
            **kwargs (Any): Additional parameters for the text generation.

        Notes:
            This method must be implemented by any subclass and should be an asynchronous operation.
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the model.

        Returns:
            Dict[str, Any]: A dictionary containing the model's name and configuration.
        """
        return {"model_name": self.model_name, "config": self.config}
