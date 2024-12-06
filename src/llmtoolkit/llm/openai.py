from collections.abc import Generator
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import PrivateAttr

from llmtoolkit.core import UNSET
from llmtoolkit.core.models import (
    ChainResponse,
    ConversationHistory,
    GenerationParameters,
    ToolCall,
    ToolCallFunction,
)
from llmtoolkit.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    _client: OpenAI = PrivateAttr()
    _async_client: AsyncOpenAI = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = OpenAI(api_key=self.api_key.get_secret_value(), base_url=self.host)
        self._async_client = AsyncOpenAI(
            api_key=self.api_key.get_secret_value(), base_url=self.host
        )

    @staticmethod
    def _prepare_tool_calls(
        tool_calls: list[ChatCompletionMessageToolCall] | None,
    ) -> list[ToolCall] | None:
        if not tool_calls:
            return None
        return [
            ToolCall(
                id=tool_call.id,
                index=getattr(tool_call, "index", None),
                type=tool_call.type,
                function=ToolCallFunction(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for tool_call in tool_calls
        ]

    def generate(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = GenerationParameters.temperature,
        top_p: float = GenerationParameters.top_p,
        frequency_penalty: float = GenerationParameters.frequency_penalty,
        presence_penalty: float = GenerationParameters.presence_penalty,
        max_tokens: int | None = GenerationParameters.max_tokens,
        stop: list[str] | None = UNSET,
        **kwargs: Any,
    ) -> ChainResponse:
        conversation_history = conversation_history or ConversationHistory()
        stop = GenerationParameters.stop if stop is UNSET else stop

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            stop=stop,
        )
        return ChainResponse(
            content=response.choices[0].message.content or "",
            metadata={
                "__tool_calls": self._prepare_tool_calls(response.choices[0].message.tool_calls)
            },
        )

    async def async_generate(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = GenerationParameters.temperature,
        top_p: float = GenerationParameters.top_p,
        frequency_penalty: float = GenerationParameters.frequency_penalty,
        presence_penalty: float = GenerationParameters.presence_penalty,
        max_tokens: int | None = GenerationParameters.max_tokens,
        stop: list[str] | None = UNSET,
        **kwargs: Any,
    ) -> ChainResponse:
        conversation_history = conversation_history or ConversationHistory()
        stop = GenerationParameters.stop if stop is UNSET else stop

        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            stop=stop,
        )
        return ChainResponse(
            content=response.choices[0].message.content or "",
            metadata={
                "__tool_calls": self._prepare_tool_calls(response.choices[0].message.tool_calls)
            },
        )

    def generate_stream(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = GenerationParameters.temperature,
        top_p: float = GenerationParameters.top_p,
        frequency_penalty: float = GenerationParameters.frequency_penalty,
        presence_penalty: float = GenerationParameters.presence_penalty,
        max_tokens: int | None = GenerationParameters.max_tokens,
        stop: list[str] | None = UNSET,
        **kwargs: Any,
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        stop = GenerationParameters.stop if stop is UNSET else stop

        for chunk in self._client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
        ):
            yield ChainResponse(
                content=chunk.choices[0].delta.content or "",
                metadata={
                    "__tool_calls": self._prepare_tool_calls(chunk.choices[0].delta.tool_calls)
                },
            )

    async def async_generate_stream(
        self,
        conversation_history: ConversationHistory | None = None,
        *,
        temperature: float = GenerationParameters.temperature,
        top_p: float = GenerationParameters.top_p,
        frequency_penalty: float = GenerationParameters.frequency_penalty,
        presence_penalty: float = GenerationParameters.presence_penalty,
        max_tokens: int | None = GenerationParameters.max_tokens,
        stop: list[str] | None = UNSET,
        **kwargs: Any,
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        stop = GenerationParameters.stop if stop is UNSET else stop

        async for chunk in await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
        ):
            yield ChainResponse(
                content=chunk.choices[0].delta.content or "",
                metadata={
                    "__tool_calls": self._prepare_tool_calls(chunk.choices[0].delta.tool_calls)
                },
            )
