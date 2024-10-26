"""
Module for implementing the OllamaLLMModel class for interacting with the Ollama language model API.

This module provides the `OllamaModel` and `OllamaAsyncModel` classes, which extend `BaseLLMModel`
and `BaseAsyncLLMModel`, respectively. These classes offer synchronous and asynchronous methods
for generating text and streaming responses using the Ollama API.
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
    A class for synchronous interactions with the Ollama language model API.

    This class extends `BaseLLMModel` to support text generation and streaming responses.

    Attributes:
        host (str): Hostname of the Ollama model.
        _client (OllamaClient): Client instance for interacting with the Ollama API.
    """

    host: str

    _client: OllamaClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = OllamaClient(host=self.host)

    def generate(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
    ) -> str:
        """
        Generates a synchronous response for the given prompt using the Ollama API.

        Args:
            prompt (str): Input text prompt for generating a response.
            conversation_history (ConversationHistory, optional): Existing conversation history.

        Returns:
            str: The generated text response.

        Raises:
            exc.OllamaConnectionError: If there is a connection error when accessing the API.
        """
        conversation_history = conversation_history or ConversationHistory()
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=conversation_history.to_dict()
                + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
                **kwargs,
            )
        except httpx.ConnectError:
            raise exc.OllamaConnectionError
        return response["message"]["content"]

    def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
    ) -> Generator[str, None, None]:
        """
        Generates a streaming response for the given prompt using the Ollama API.

        Args:
            prompt (str): Input text prompt for generating a streaming response.
            conversation_history (ConversationHistory, optional): Existing conversation history.

        Yields:
            str: Streaming chunks of the response text.

        Raises:
            exc.OllamaConnectionError: If there is a connection error when accessing the API.
        """
        conversation_history = conversation_history or ConversationHistory()
        try:
            for chunk in self._client.chat(
                model=self.model_name,
                messages=conversation_history.to_dict()
                + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
                stream=True,
                **kwargs,
            ):
                yield chunk["message"]["content"]
        except httpx.ConnectError:
            raise exc.OllamaConnectionError


class OllamaAsyncModel(BaseAsyncLLMModel):
    """
    A class for asynchronous interactions with the Ollama language model API.

    This class extends `BaseAsyncLLMModel` to support asynchronous text generation and
    streaming responses.

    Attributes:
        host (str): Hostname of the Ollama model.
        _client (OllamaAsyncClient): Asynchronous client instance for interacting
        with the Ollama API.
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
            prompt (str): Input text prompt for generating a response.
            conversation_history (ConversationHistory, optional): Existing conversation history.

        Returns:
            str: The generated text response.

        Raises:
            exc.OllamaConnectionError: If there is a connection error when accessing the API.
        """
        conversation_history = conversation_history or ConversationHistory()
        try:
            response = await self._client.chat(
                model=self.model_name,
                messages=conversation_history.to_dict()
                + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
                **kwargs,
            )
        except httpx.ConnectError:
            raise exc.OllamaConnectionError
        return response["message"]["content"]

    async def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
    ) -> Generator[str, None, None]:
        """
        Asynchronously generates a streaming response for the given prompt using the Ollama API.

        Args:
            prompt (str): Input text prompt for generating a streaming response.
            conversation_history (ConversationHistory, optional): Existing conversation history.

        Yields:
            str: Streaming chunks of the response text.

        Raises:
            exc.OllamaConnectionError: If there is a connection error when accessing the API.
        """
        conversation_history = conversation_history or ConversationHistory()
        try:
            async for chunk in await self._client.chat(
                model=self.model_name,
                messages=conversation_history.to_dict()
                + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
                stream=True,
                **kwargs,
            ):
                yield chunk["message"]["content"]
        except httpx.ConnectError:
            raise exc.OllamaConnectionError
