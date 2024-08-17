"""
Conversation - provides interaction with a language model.
This module defines the `Conversation` class for handling chat and streaming interactions with a
language model, including support for conversation history.
"""

from collections.abc import Generator

from .base import BaseConversation
from .history import ConversationMessage


class Conversation(BaseConversation):
    """
    Extends `BaseConversation` to provide chat and streaming capabilities.
    """

    async def chat(self, prompt: str) -> str:
        """
        Generates a response from the language model.

        Args:
            prompt (str): The prompt for the model.

        Returns:
            Response: The model's response.
        """
        response = await self.llm.generate(prompt, self.history.copy(deep=True))
        self.history.extend(
            [
                ConversationMessage(role="user", content=prompt),
                ConversationMessage(role="assistant", content=response),
            ]
        )
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
            AsyncIterator[Response]: An iterator of the model's responses.
        """
        return self.llm.generate_stream(prompt, self.history)
