from collections.abc import Generator
from typing import Any

import httpx
from mistralai import Mistral as MistralClient
from mistralai import ToolCall as MistralToolCall
from pydantic import PrivateAttr, SecretStr

from llmtoolkit.core.models import ChainResponse, ConversationHistory, ToolCall, ToolCallFunction
from llmtoolkit.exc import StreamReadError
from llmtoolkit.llm.base import BaseLLM


class MistralaiLLM(BaseLLM):
    api_key: SecretStr

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

    def generate(self, conversation_history: ConversationHistory | None = None, **kwargs) -> str:
        conversation_history = conversation_history or ConversationHistory()
        response = self._client.chat.complete(
            model=self.model_name, messages=conversation_history.dump(), **kwargs
        )
        return ChainResponse(
            content=response.choices[0].message.content,
            metadata={
                "__tool_calls": self._prepare_tool_calls(response.choices[0].message.tool_calls)
            },
        )

    async def async_generate(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> str:
        conversation_history = conversation_history or ConversationHistory()
        response = await self._client.chat.complete_async(
            model=self.model_name, messages=conversation_history.dump(), **kwargs
        )
        return ChainResponse(
            content=response.choices[0].message.content,
            metadata={
                "__tool_calls": self._prepare_tool_calls(response.choices[0].message.tool_calls)
            },
        )

    def generate_stream(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> Generator[str, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        try:
            for chunk in self._client.chat.stream(
                model=self.model_name,
                messages=conversation_history.dump(),
                **kwargs,
            ):
                yield ChainResponse(
                    content=chunk.data.choices[0].delta.content or "",
                    metadata={
                        "__tool_calls": self._prepare_tool_calls(
                            chunk.data.choices[0].delta.tool_calls
                        )
                    },
                )
        except httpx.ResponseNotRead:
            raise StreamReadError

    async def async_generate_stream(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> Generator[str, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        try:
            async for chunk in await self._client.chat.stream_async(
                model=self.model_name,
                messages=conversation_history.dump(),
                **kwargs,
            ):
                yield ChainResponse(
                    content=chunk.data.choices[0].delta.content or "",
                    metadata={
                        "__tool_calls": self._prepare_tool_calls(
                            chunk.data.choices[0].delta.tool_calls
                        )
                    },
                )
        except httpx.ResponseNotRead:
            raise StreamReadError
