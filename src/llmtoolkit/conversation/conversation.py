"""
Conversation - provides interaction with a language model.
This module defines the `Conversation` class for handling chat and streaming interactions with a
language model, including support for conversation history.
"""

from typing import Generator

from llmtoolkit.conversation.base import BaseConversation
from llmtoolkit.conversation.history import ConversationHistory


class Conversation(BaseConversation):
    """
    Extends `BaseConversation` to provide chat and streaming capabilities.
    """

    async def chat(
        self, prompt: str, conversation_history: ConversationHistory = None
    ) -> str:
        """
        Generates a response from the language model.

        Args:
            prompt (str): The prompt for the model.
            conversation_history (ConversationHistory, optional): The conversation history.

        Returns:
            Response: The model's response.
        """
        return await self.llm.generate(prompt, conversation_history)

    async def stream(
        self, prompt: str, conversation_history: ConversationHistory = None
    ) -> Generator[str, None, None]:
        """
        Streams responses from the language model.

        Args:
            prompt (str): The prompt for the model.
            conversation_history (ConversationHistory, optional): The conversation history.

        Returns:
            AsyncIterator[Response]: An iterator of the model's responses.
        """
        return await self.llm.generate_stream(prompt, conversation_history)
