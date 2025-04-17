from collections.abc import AsyncGenerator, Generator
from typing import Any

from openai import NOT_GIVEN, AsyncOpenAI, OpenAI
from pydantic import PrivateAttr

from llmtoolkit.core import BaseLLM
from llmtoolkit.core.models import (
    ChainResponse,
    ConversationHistory,
)


class OpenAILLM(BaseLLM):
    _client: OpenAI = PrivateAttr()
    _async_client: AsyncOpenAI = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.host)
        self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.host)

    def generate(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = NOT_GIVEN,
        top_p: float = NOT_GIVEN,
        frequency_penalty: float = NOT_GIVEN,
        presence_penalty: float = NOT_GIVEN,
        max_completion_tokens: int | None = NOT_GIVEN,
        stop: list[str] | None = NOT_GIVEN,
        **kwargs: Any,
    ) -> ChainResponse:
        conversation_history = conversation_history or ConversationHistory()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            **kwargs,
        )
        message = response.choices[0].message
        return ChainResponse(
            content=message.content or "",
        )

    async def agenerate(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = NOT_GIVEN,
        top_p: float = NOT_GIVEN,
        frequency_penalty: float = NOT_GIVEN,
        presence_penalty: float = NOT_GIVEN,
        max_completion_tokens: int | None = NOT_GIVEN,
        stop: list[str] | None = NOT_GIVEN,
        **kwargs: Any,
    ) -> ChainResponse:
        conversation_history = conversation_history or ConversationHistory()
        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            **kwargs,
        )
        message = response.choices[0].message
        return ChainResponse(
            content=message.content or "",
        )

    def stream(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = NOT_GIVEN,
        top_p: float = NOT_GIVEN,
        frequency_penalty: float = NOT_GIVEN,
        presence_penalty: float = NOT_GIVEN,
        max_completion_tokens: int | None = NOT_GIVEN,
        stop: list[str] | None = NOT_GIVEN,
        **kwargs: Any,
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            yield ChainResponse(
                content=delta.content or "",
            )

    async def astream(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = NOT_GIVEN,
        top_p: float = NOT_GIVEN,
        frequency_penalty: float = NOT_GIVEN,
        presence_penalty: float = NOT_GIVEN,
        max_completion_tokens: int | None = NOT_GIVEN,
        stop: list[str] | None = NOT_GIVEN,
        **kwargs: Any,
    ) -> AsyncGenerator[ChainResponse, None]:
        conversation_history = conversation_history or ConversationHistory()
        async for chunk in await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            stream=True,
            **kwargs,
        ):
            delta = chunk.choices[0].delta
            yield ChainResponse(
                content=delta.content or "",
            )
