"""
Conversation - provides interaction with a language model.
This module defines the `Conversation` class for handling chat
and streaming interactions with a language model, including
support for conversation history.
"""

from collections.abc import Generator

from .base import BaseAsyncConversation, BaseConversation
from .history import ConversationMessage, Roles


class Conversation(BaseConversation):
    """
    Extends `BaseConversation` to provide chat and streaming capabilities.
    """

    def chat(self, prompt: str) -> str:
        """
        Generates a response from the language model.

        Args:
            prompt (str): The prompt for the model.

        Returns:
            Response: The model's response.
        """
        self.history.append(ConversationMessage(role=Roles.USER, content=prompt))
        response = self.llm.generate(prompt, self.history)
        self.history.append(ConversationMessage(role=Roles.ASSISTANT, content=response))
        return response

    def stream(
        self,
        prompt: str,
    ) -> Generator[str, None, None]:
        """
        Streams responses from the language model.

        Args:
            prompt (str): The prompt for the model.

        Returns:
            Generator[Response]: An iterator of the model's responses.
        """
        self.history.extend(
            [
                ConversationMessage(role=Roles.USER, content=prompt),
                ConversationMessage(role=Roles.ASSISTANT, content=""),
            ]
        )
        return self.llm.generate_stream(prompt, self.history)


class AsyncConversation(BaseAsyncConversation):
    """
    Extends `BaseAsyncConversation` to provide chat and streaming capabilities.
    """

    async def chat(self, prompt: str) -> str:
        """
        Generates a response from the language model.

        Args:
            prompt (str): The prompt for the model.

        Returns:
            Response: The model's response.
        """
        self.history.append(ConversationMessage(role=Roles.USER, content=prompt))
        response = await self.llm.generate(prompt, self.history)
        self.history.append(ConversationMessage(role=Roles.ASSISTANT, content=response))
        return response

    async def stream(
        self,
        prompt: str,
    ) -> Generator[str, None, None]:
        """
        Streams responses from the language model.

        Args:
            prompt (str): The prompt for the model.

        Returns:
            Generator[Response]: An iterator of the model's responses.
        """
        self.history.extend(
            [
                ConversationMessage(role=Roles.USER, content=prompt),
                ConversationMessage(role=Roles.ASSISTANT, content=""),
            ]
        )
        return self.llm.generate_stream(prompt, self.history)
