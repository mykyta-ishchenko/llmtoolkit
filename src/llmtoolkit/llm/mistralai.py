from collections.abc import Generator
from typing import Any

import httpx
from mistralai import Mistral as MistralClient
from mistralai import ToolCall as MistralToolCall
from pydantic import PrivateAttr

from llmtoolkit.core import UNSET
from llmtoolkit.core.models import (
    ChainResponse,
    ConversationHistory,
    GenerationParameters,
    ToolCall,
    ToolCallFunction,
)
from llmtoolkit.exc import StreamReadError
from llmtoolkit.llm.base import BaseLLM


class MistralaiLLM(BaseLLM):
    _client: MistralClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = MistralClient(api_key=self.api_key.get_secret_value())

    @staticmethod
    def _prepare_tool_calls(tool_calls: list[MistralToolCall] | None) -> list[ToolCall] | None:
        if not tool_calls:
            return None
        return [
            ToolCall(
                id=tool_call.id,
                function=ToolCallFunction(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for tool_call in tool_calls
        ]

    def _make_chain_response(
        self, content: str, tool_calls: list[MistralToolCall] | None
    ) -> ChainResponse:
        return ChainResponse(
            content=content, metadata={"__tool_calls": self._prepare_tool_calls(tool_calls)}
        )

    def _extract_content_and_tool_calls(
        self, response: Any
    ) -> tuple[str, list[MistralToolCall] | None]:
        try:
            choice = response.choices[0]
            message = choice.message if hasattr(choice, "message") else choice.delta
            content = message.content or ""
            tool_calls = message.tool_calls
        except (IndexError, AttributeError):
            content = ""
            tool_calls = None
        return content, tool_calls

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

        response = self._client.chat.complete(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            stop=stop,
        )

        content, tool_calls = self._extract_content_and_tool_calls(response)
        return self._make_chain_response(content, tool_calls)

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

        response = await self._client.chat.complete_async(
            model=self.model_name,
            messages=conversation_history.dump(),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            stop=stop,
        )

        content, tool_calls = self._extract_content_and_tool_calls(response)
        return self._make_chain_response(content, tool_calls)

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

        try:
            for chunk in self._client.chat.stream(
                model=self.model_name,
                messages=conversation_history.dump(),
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                stop=stop,
            ):
                content, tool_calls = self._extract_content_and_tool_calls(chunk.data)
                yield self._make_chain_response(content, tool_calls)
        except httpx.ResponseNotRead:
            raise StreamReadError

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

        try:
            async for chunk in await self._client.chat.stream_async(
                model=self.model_name,
                messages=conversation_history.dump(),
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                stop=stop,
            ):
                content, tool_calls = self._extract_content_and_tool_calls(chunk.data)
                yield self._make_chain_response(content, tool_calls)
        except httpx.ResponseNotRead:
            raise StreamReadError
