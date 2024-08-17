"""
Module for implementing the OllamaLLMModel class for interacting with the Ollama language model API.

This module contains the `OllamaLLMModel` class, which extends the `BaseLLMModel` to provide
asynchronous methods for generating text and streaming responses using the Ollama API.
"""

from collections.abc import Generator

from ollama import AsyncClient as OllamaAsyncClient

from llmtoolkit.conversation import ConversationHistory, ConversationMessage
from llmtoolkit.llms.base import BaseLLMModel


class OllamaLLMModel(BaseLLMModel):
    """
    A class for interacting with the Ollama language model API.

    Inherits from `BaseLLMModel` and provides implementation for generating text and
    streaming responses asynchronously.

    Attributes:
        _client (OllamaAsyncClient): The asynchronous client for interacting with the Ollama API.
    """

    def __init__(self, model_name: str, host: str):
        """
        Initializes the OllamaLLMModel instance with the model name and API host.

        Args:
            model_name (str): The name of the model to be used with the Ollama API.
            host (str): The host URL for the Ollama API.
        """
        self._client = OllamaAsyncClient(host=host)
        super().__init__(model_name)

    def __prepare_conversation_history(
        self, prompt: str, conversation_history: ConversationHistory
    ) -> ConversationHistory:
        """
        Prepares the conversation history for the current prompt by
            appending the prompt as a new message.

        Args:
            prompt (str): The input text prompt for the model.
            conversation_history (ConversationHistory): The existing conversation history.

        Returns:
            ConversationHistory: The updated conversation history with the new prompt added.
        """
        conversation_history = conversation_history or ConversationHistory()
        conversation_history.append(ConversationMessage(role="user", content=prompt))
        return conversation_history

    async def generate(
        self,
        prompt: str,
        conversation_history: ConversationHistory = None,
    ) -> str:
        """
        Asynchronously generates a response for the given prompt using the Ollama API.

        Args:
            prompt (str): The input text prompt for the model.
            conversation_history (ConversationHistory, optional): The existing conversation history.

        Returns:
            str: The generated text response.

        Raises:
            Exception: If there is an error in generating the response from the API.
        """
        conversation_history = self.__prepare_conversation_history(prompt, conversation_history)
        response = await self._client.chat(
            model=self.model_name,
            messages=conversation_history.model_dump().get("messages"),
        )
        return response["message"]["content"]

    async def generate_stream(
        self,
        prompt: str,
        conversation_history: ConversationHistory = None,
    ) -> Generator[str, None, None]:
        """
        Asynchronously generates a response in a streaming manner for
            the given prompt using the Ollama API.

        Args:
            prompt (str): The input text prompt for the model.
            conversation_history (ConversationHistory, optional): The existing conversation history.

        Yields:
            str: The streaming text response chunks.

        Raises:
            Exception: If there is an error in streaming the response from the API.
        """
        conversation_history = self.__prepare_conversation_history(prompt, conversation_history)
        async for chunk in await self._client.chat(
            model=self.model_name,
            messages=conversation_history.model_dump().get("messages"),
            stream=True,
        ):
            yield chunk["message"]["content"]
