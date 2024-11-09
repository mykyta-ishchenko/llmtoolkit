import json
from collections.abc import Generator
from typing import Any

from ollama import AsyncClient as OllamaAsyncClient
from ollama import Client as OllamaClient
from pydantic import PrivateAttr

from llmtoolkit.core.models import ChainResponse, ConversationHistory, ToolCall, ToolCallFunction

from .base import BaseLLM


class OllamaLLM(BaseLLM):
    host: str

    _client: OllamaClient = PrivateAttr()
    _async_client: OllamaAsyncClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = OllamaClient(host=self.host)
        self._async_client = OllamaAsyncClient(host=self.host)

    @staticmethod
    def _prepare_tool_calls(tool_calls: dict[str, Any] | None) -> list[ToolCall] | None:
        if not tool_calls:
            return None
        return [
            ToolCall(
                function=ToolCallFunction(
                    name=tool_call["function"]["name"],
                    arguments=json.dumps(tool_call["function"]["arguments"]),
                ),
            )
            for tool_call in tool_calls
        ]

    def generate(
        self,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> ChainResponse:
        conversation_history = conversation_history or ConversationHistory()
        response = self._client.chat(
            model=self.model_name,
            messages=conversation_history.dump(),
            **kwargs,
        )
        return ChainResponse(
            content=response["message"]["content"],
            metadata={
                "__tool_calls": self._prepare_tool_calls(response["message"].get("tool_calls"))
            },
        )

    async def async_generate(
        self,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> str:
        conversation_history = conversation_history or ConversationHistory()
        response = await self._async_client.chat(
            model=self.model_name,
            messages=conversation_history.dump(),
            **kwargs,
        )
        return ChainResponse(
            content=response["message"]["content"],
            metadata={
                "__tool_calls": self._prepare_tool_calls(response["message"].get("tool_calls"))
            },
        )

    def generate_stream(
        self,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        for chunk in self._client.chat(
            model=self.model_name,
            messages=conversation_history.dump(),
            stream=True,
            **kwargs,
        ):
            yield ChainResponse(
                content=chunk["message"]["content"],
                metadata={
                    "__tool_calls": self._prepare_tool_calls(chunk["message"].get("tool_calls"))
                },
            )

    async def async_generate_stream(
        self,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        async for chunk in await self._async_client.chat(
            model=self.model_name,
            messages=conversation_history.dump(),
            stream=True,
            **kwargs,
        ):
            yield ChainResponse(
                content=chunk["message"]["content"],
                metadata={
                    "__tool_calls": self._prepare_tool_calls(chunk["message"].get("tool_calls"))
                },
            )
