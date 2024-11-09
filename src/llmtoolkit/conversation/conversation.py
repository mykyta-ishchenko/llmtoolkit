from collections.abc import Generator

from llmtoolkit.core.models import ChainResponse, ConversationMessage, Roles

from .base import BaseConversation


class Conversation(BaseConversation):
    def chat(self, prompt: str, **kwargs) -> ChainResponse:
        self.history.append(ConversationMessage(role=Roles.USER, content=prompt))
        response = self.chain.generate(self.history.copy(deep=True), **kwargs)
        self.history.append(ConversationMessage(role=Roles.ASSISTANT, content=response.content))
        return response

    async def async_chat(self, prompt: str, **kwargs) -> ChainResponse:
        self.history.append(ConversationMessage(role=Roles.USER, content=prompt))
        response = await self.chain.async_generate(self.history.copy(deep=True), **kwargs)
        self.history.append(ConversationMessage(role=Roles.ASSISTANT, content=response.content))
        return response

    def stream(self, prompt: str, **kwargs) -> Generator[ChainResponse, None, None]:
        self.history.append(ConversationMessage(role=Roles.USER, content=prompt))
        assistant = ""
        for chunk in self.chain.generate_stream(self.history.copy(deep=True), **kwargs):
            assistant += chunk.content
            yield chunk
        self.history.append(ConversationMessage(role=Roles.ASSISTANT, content=assistant))

    async def async_stream(self, prompt: str, **kwargs) -> Generator[ChainResponse, None, None]:
        self.history.append(ConversationMessage(role=Roles.USER, content=prompt))
        assistant = ""
        async for chunk in await self.chain.async_generate_stream(
            self.history.copy(deep=True), **kwargs
        ):
            assistant += chunk.content
            yield chunk
        self.history.append(ConversationMessage(role=Roles.ASSISTANT, content=assistant))
