"""
Conversation - provides interaction with a language model.

This module defines the `Conversation` and `AsyncConversation` classes for handling
synchronous and asynchronous chat and streaming interactions with a language model,
including support for maintaining conversation history.
"""

from collections.abc import Generator

from .base import BaseAsyncConversation, BaseConversation
from .history import ConversationMessage, Roles


class Conversation(BaseConversation):
    """
    Extends `BaseConversation` to provide synchronous chat and streaming capabilities
    with conversation history support.

    Attributes:
        history (ConversationHistory): Tracks the conversation history, including user prompts
            and model responses.
        llm (Any): Synchronous language model instance used for generating responses.
    """

    def chat(self, prompt: str, **kwargs) -> str:
        """
        Generates a synchronous response from the language model based on the given prompt.

        Args:
            prompt (str): Input prompt for generating a response.

        Returns:
            str: The generated response from the model.
        """
        self.history.append(ConversationMessage(role=Roles.USER, content=prompt))
        response = self.llm.generate(prompt, self.history, **kwargs)
        self.history.append(ConversationMessage(role=Roles.ASSISTANT, content=response))
        return response

    def stream(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Streams synchronous responses from the language model based on the given prompt.

        Args:
            prompt (str): Input prompt for generating a streaming response.

        Yields:
            str: Chunks of the model's response streamed iteratively.
        """
        self.history.extend(
            [
                ConversationMessage(role=Roles.USER, content=prompt),
                ConversationMessage(role=Roles.ASSISTANT, content=""),
            ]
        )
        return self.llm.generate_stream(prompt, self.history, **kwargs)


class AsyncConversation(BaseAsyncConversation):
    """
    Extends `BaseAsyncConversation` to provide asynchronous chat and streaming capabilities
    with conversation history support.

    Attributes:
        history (ConversationHistory): Tracks the conversation history, including user prompts
            and model responses.
        llm (Any): Asynchronous language model instance used for generating responses.
    """

    async def chat(self, prompt: str, **kwargs) -> str:
        """
        Asynchronously generates a response from the language model based on the given prompt.

        Args:
            prompt (str): Input prompt for generating a response.

        Returns:
            str: The generated response from the model.
        """
        self.history.append(ConversationMessage(role=Roles.USER, content=prompt))
        response = await self.llm.generate(prompt, self.history, **kwargs)
        self.history.append(ConversationMessage(role=Roles.ASSISTANT, content=response))
        return response

    async def stream(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Streams asynchronous responses from the language model based on the given prompt.

        Args:
            prompt (str): Input prompt for generating a streaming response.

        Yields:
            str: Chunks of the model's response streamed iteratively.
        """
        self.history.extend(
            [
                ConversationMessage(role=Roles.USER, content=prompt),
                ConversationMessage(role=Roles.ASSISTANT, content=""),
            ]
        )
        return self.llm.generate_stream(prompt, self.history, **kwargs)
