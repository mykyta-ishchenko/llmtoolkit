from collections.abc import Generator
from typing import Any

from pydantic import Field

from llmtoolkit.core import UNSET
from llmtoolkit.core.models import ChainResponse, ConversationMessage, GenerationParameters, Roles

from .base import BaseConversation


class Conversation(BaseConversation):
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)

    def _add_user_message(self, content: str) -> None:
        self.history.append(ConversationMessage(role=Roles.USER, content=content))

    def _append_assistant_message(self, content: str) -> None:
        self.history.append(ConversationMessage(role=Roles.ASSISTANT, content=content))

    def chat(
        self,
        prompt: str,
        *,
        temperature: float = UNSET,
        top_p: float = UNSET,
        frequency_penalty: float = UNSET,
        presence_penalty: float = UNSET,
        max_tokens: int | None = UNSET,
        stop: list[str] | None = UNSET,
        **kwargs: Any,
    ) -> ChainResponse:
        self._add_user_message(prompt)
        response = self.chain.generate(
            conversation_history=self.history.copy(deep=True),
            temperature=self.parameters.temperature if temperature is UNSET else temperature,
            top_p=self.parameters.top_p if top_p is UNSET else top_p,
            frequency_penalty=self.parameters.frequency_penalty
            if frequency_penalty is UNSET
            else frequency_penalty,
            presence_penalty=self.parameters.presence_penalty
            if presence_penalty is UNSET
            else presence_penalty,
            max_tokens=self.parameters.max_tokens if max_tokens is UNSET else max_tokens,
            stop=self.parameters.stop if stop is UNSET else stop,
            **kwargs,
        )
        self._append_assistant_message(response.content)
        return response

    async def async_chat(
        self,
        prompt: str,
        *,
        temperature: float = UNSET,
        top_p: float = UNSET,
        frequency_penalty: float = UNSET,
        presence_penalty: float = UNSET,
        max_tokens: int | None = UNSET,
        stop: list[str] | None = UNSET,
        **kwargs: Any,
    ) -> ChainResponse:
        self._add_user_message(prompt)
        response = await self.chain.async_generate(
            conversation_history=self.history.copy(deep=True),
            temperature=self.parameters.temperature if temperature is UNSET else temperature,
            top_p=self.parameters.top_p if top_p is UNSET else top_p,
            frequency_penalty=self.parameters.frequency_penalty
            if frequency_penalty is UNSET
            else frequency_penalty,
            presence_penalty=self.parameters.presence_penalty
            if presence_penalty is UNSET
            else presence_penalty,
            max_tokens=self.parameters.max_tokens if max_tokens is UNSET else max_tokens,
            stop=self.parameters.stop if stop is UNSET else stop,
            **kwargs,
        )
        self._append_assistant_message(response.content)
        return response

    def stream(
        self,
        prompt: str,
        *,
        temperature: float = UNSET,
        top_p: float = UNSET,
        frequency_penalty: float = UNSET,
        presence_penalty: float = UNSET,
        max_tokens: int | None = UNSET,
        stop: list[str] | None = UNSET,
        **kwargs: Any,
    ) -> Generator[ChainResponse, None, None]:
        self._add_user_message(prompt)
        assistant_content = []
        for chunk in self.chain.generate_stream(
            conversation_history=self.history.copy(deep=True),
            temperature=self.parameters.temperature if temperature is UNSET else temperature,
            top_p=self.parameters.top_p if top_p is UNSET else top_p,
            frequency_penalty=self.parameters.frequency_penalty
            if frequency_penalty is UNSET
            else frequency_penalty,
            presence_penalty=self.parameters.presence_penalty
            if presence_penalty is UNSET
            else presence_penalty,
            max_tokens=self.parameters.max_tokens if max_tokens is UNSET else max_tokens,
            stop=self.parameters.stop if stop is UNSET else stop,
            **kwargs,
        ):
            assistant_content.append(chunk.content)
            yield chunk
        self._append_assistant_message("".join(assistant_content))

    async def async_stream(
        self,
        prompt: str,
        *,
        temperature: float = UNSET,
        top_p: float = UNSET,
        frequency_penalty: float = UNSET,
        presence_penalty: float = UNSET,
        max_tokens: int | None = UNSET,
        stop: list[str] | None = UNSET,
        **kwargs: Any,
    ) -> Generator[ChainResponse, None, None]:
        self._add_user_message(prompt)
        assistant_content = []
        async for chunk in await self.chain.async_generate_stream(
            conversation_history=self.history.copy(deep=True),
            temperature=self.parameters.temperature if temperature is UNSET else temperature,
            top_p=self.parameters.top_p if top_p is UNSET else top_p,
            frequency_penalty=self.parameters.frequency_penalty
            if frequency_penalty is UNSET
            else frequency_penalty,
            presence_penalty=self.parameters.presence_penalty
            if presence_penalty is UNSET
            else presence_penalty,
            max_tokens=self.parameters.max_tokens if max_tokens is UNSET else max_tokens,
            stop=self.parameters.stop if stop is UNSET else stop,
            **kwargs,
        ):
            assistant_content.append(chunk.content)
            yield chunk
        self._append_assistant_message("".join(assistant_content))
