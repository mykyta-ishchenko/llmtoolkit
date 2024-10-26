"""
Module for implementing the OllamaLLMModel class for interacting with the Ollama language model API.

This module contains the `OllamaLLMModel` class, which extends the `BaseLLMModel` to provide
asynchronous methods for generating text and streaming responses using the Ollama API.
"""

from collections.abc import Generator
from typing import Any

import httpx
from ollama import AsyncClient as OllamaAsyncClient
from ollama import Client as OllamaClient
from pydantic import PrivateAttr

import llmtoolkit.exc as exc
from llmtoolkit.conversation import ConversationHistory, ConversationMessage, Roles

from .base import BaseAsyncLLMModel, BaseLLMModel


class OllamaModel(BaseLLMModel):
    """
    A class for interacting with the Ollama language model API.

    Inherits from `BaseLLMModel` and provides implementation for generating text and
    streaming responses.

    Attributes:
        host: The host  name of the Ollama model.
        _client (OllamaClient): The client for interacting with the Ollama API.
    """

    host: str

    _client: OllamaClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = OllamaClient(host=self.host)

    def generate(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
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
        conversation_history = conversation_history or ConversationHistory()
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=conversation_history.to_dict()
                + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
            )
        except httpx.ConnectError:
            raise exc.OllamaConnectionError
        return response["message"]["content"]

    def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
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
        conversation_history = conversation_history or ConversationHistory()
        try:
            for chunk in self._client.chat(
                model=self.model_name,
                messages=conversation_history.to_dict()
                + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
                stream=True,
            ):
                yield chunk["message"]["content"]
        except httpx.ConnectError:
            raise exc.OllamaConnectionError


class OllamaAsyncModel(BaseAsyncLLMModel):
    """
    A class for interacting with the Ollama language model API.

    Inherits from `BaseLLMModel` and provides implementation for generating text and
    streaming responses asynchronously.

    Attributes:
        _client (OllamaAsyncClient): The asynchronous client for interacting with the Ollama API.
    """

    host: str

    _client: OllamaAsyncClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = OllamaAsyncClient(host=self.host)

    async def generate(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
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
        conversation_history = conversation_history or ConversationHistory()
        try:
            response = await self._client.chat(
                model=self.model_name,
                messages=conversation_history.to_dict()
                + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
            )
        except httpx.ConnectError:
            raise exc.OllamaConnectionError
        return response["message"]["content"]

    async def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
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
        conversation_history = conversation_history or ConversationHistory()
        try:
            async for chunk in await self._client.chat(
                model=self.model_name,
                messages=conversation_history.to_dict()
                + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
                stream=True,
            ):
                yield chunk["message"]["content"]
        except httpx.ConnectError:
            raise exc.OllamaConnectionError
