"""
Module for implementing the MistralLLMModel class for interacting with the
Mistral language model API.

This module provides the `MistralModel` and `MistralAsyncModel` classes, which extend `BaseLLMModel`
and `BaseAsyncLLMModel` respectively. These classes offer synchronous and asynchronous methods
for generating text and streaming responses using the Mistral API.
"""

from collections.abc import Generator
from typing import Any

from mistralai import Mistral as MistralClient
from pydantic import PrivateAttr

from llmtoolkit.conversation import ConversationHistory, ConversationMessage, Roles

from .base import BaseAsyncLLMModel, BaseLLMModel


class MistralModel(BaseLLMModel):
    """
    A class for synchronous interactions with the Mistral language model API.

    This class extends `BaseLLMModel` to support text generation and streaming responses.

    Attributes:
        api_key: API key for authenticating requests.
        _client (MistralClient): Client instance for interacting with the Mistral API.
    """

    api_key: str

    _client: MistralClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = MistralClient(api_key=self.api_key)

    def generate(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
    ) -> str:
        """
        Generates a synchronous response for a given prompt using the Mistral API.

        Args:
            prompt (str): Input prompt for generating a response.
            conversation_history (ConversationHistory, optional): Existing conversation history.

        Returns:
            str: The generated text response.

        Raises:
            Exception: If an error occurs during response generation.
        """
        conversation_history = conversation_history or ConversationHistory()
        response = self._client.chat.complete(
            model=self.model_name,
            messages=conversation_history.to_dict()
            + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
        )
        return response.choices[0].message.content

    def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
    ) -> Generator[str, None, None]:
        """
        Generates a streaming response for a given prompt using the Mistral API.

        Args:
            prompt (str): Input prompt for generating a streaming response.
            conversation_history (ConversationHistory, optional): Existing conversation history.

        Yields:
            str: Streaming chunks of the response text.

        Raises:
            Exception: If an error occurs during response streaming.
        """
        conversation_history = conversation_history or ConversationHistory()
        for chunk in self._client.chat.stream(
            model=self.model_name,
            messages=conversation_history.to_dict()
            + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
            **kwargs,
        ):
            yield chunk.data.choices[0].delta.content


class MistralAsyncModel(BaseAsyncLLMModel):
    """
    A class for asynchronous interactions with the Mistral language model API.

    This class extends `BaseAsyncLLMModel` to support asynchronous text generation and
    streaming responses.

    Attributes:
        api_key: API key for authenticating requests.
        _client (MistralClient): Asynchronous client instance for interacting with the Mistral API.
    """

    api_key: str

    _client: MistralClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = MistralClient(api_key=self.api_key)

    async def generate(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
    ) -> str:
        """
        Asynchronously generates a response for a given prompt using the Mistral API.

        Args:
            prompt (str): Input prompt for generating a response.
            conversation_history (ConversationHistory, optional): Existing conversation history.

        Returns:
            str: The generated text response.

        Raises:
            Exception: If an error occurs during response generation.
        """
        conversation_history = conversation_history or ConversationHistory()
        response = await self._client.chat.complete_async(
            model=self.model_name,
            messages=conversation_history.to_dict()
            + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
            **kwargs,
        )
        return response.choices[0].message.content

    async def generate_stream(
        self, prompt: str, conversation_history: ConversationHistory = None, **kwargs
    ) -> Generator[str, None, None]:
        """
        Asynchronously generates a streaming response for a given prompt using the Mistral API.

        Args:
            prompt (str): Input prompt for generating a streaming response.
            conversation_history (ConversationHistory, optional): Existing conversation history.

        Yields:
            str: Streaming chunks of the response text.

        Raises:
            Exception: If an error occurs during response streaming.
        """
        conversation_history = conversation_history or ConversationHistory()
        async for chunk in await self._client.chat.stream_async(
            model=self.model_name,
            messages=conversation_history.to_dict()
            + [ConversationMessage(role=Roles.USER, content=prompt).model_dump()],
            **kwargs,
        ):
            yield chunk.data.choices[0].delta.content
