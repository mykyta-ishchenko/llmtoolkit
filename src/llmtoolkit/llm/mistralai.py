from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx
from mistralai import UNSET as MISTRAL_UNSET
from mistralai import Mistral as MistralClient
from pydantic import PrivateAttr

from llmtoolkit.core.llm import BaseLLM
from llmtoolkit.core.models import (
    ChainResponse,
    ConversationHistory,
)
from llmtoolkit.exc import StreamReadError


class MistralaiLLM(BaseLLM):
    _client: MistralClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = MistralClient(api_key=self.api_key)

    def generate(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = MISTRAL_UNSET,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        max_completion_tokens: int | None = MISTRAL_UNSET,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChainResponse:
        conversation_history = conversation_history or ConversationHistory()
        response = self._client.chat.complete(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_completion_tokens,
            stop=stop,
            **kwargs,
        )
        message = response.choices[0].message
        return ChainResponse(content=message.content or "")

    async def agenerate(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = MISTRAL_UNSET,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        max_completion_tokens: int | None = MISTRAL_UNSET,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChainResponse:
        conversation_history = conversation_history or ConversationHistory()
        response = await self._client.chat.complete_async(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_completion_tokens,
            stop=stop,
            **kwargs,
        )
        message = response.choices[0].message
        return ChainResponse(content=message.content or "")

    def stream(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = MISTRAL_UNSET,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        max_completion_tokens: int | None = MISTRAL_UNSET,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        try:
            for chunk in self._client.chat.stream(
                model=self.model_name,
                messages=conversation_history.dump(),
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_completion_tokens,
                stop=stop,
                **kwargs,
            ):
                message = chunk.data.choices[0].delta
                yield ChainResponse(content=message.content or "")
        except httpx.ResponseNotRead:
            raise StreamReadError

    async def astream(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = MISTRAL_UNSET,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        max_completion_tokens: int | None = MISTRAL_UNSET,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChainResponse, None]:
        conversation_history = conversation_history or ConversationHistory()
        try:
            async for chunk in await self._client.chat.stream_async(
                model=self.model_name,
                messages=conversation_history.dump(),
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_completion_tokens,
                stop=stop,
                **kwargs,
            ):
                message = chunk.data.choices[0].delta
                yield ChainResponse(content=message.content or "")
        except httpx.ResponseNotRead:
            raise StreamReadError
