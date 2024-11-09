from collections.abc import Generator
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import PrivateAttr, SecretStr

from llmtoolkit.core.models import ChainResponse, ConversationHistory, ToolCall, ToolCallFunction
from llmtoolkit.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    api_key: SecretStr
    base_url: str | None = None

    _client: OpenAI = PrivateAttr()
    _async_client: AsyncOpenAI = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = OpenAI(api_key=self.api_key.get_secret_value(), base_url=self.base_url)
        self._async_client = AsyncOpenAI(
            api_key=self.api_key.get_secret_value(), base_url=self.base_url
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
                index=tool_call.index if hasattr(tool_call, "index") else None,
                type=tool_call.type,
                function=ToolCallFunction(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for tool_call in tool_calls
        ]

    def generate(self, conversation_history: ConversationHistory | None = None, **kwargs) -> str:
        conversation_history = conversation_history or ConversationHistory()
        response = self._client.chat.completions.create(
            model=self.model_name, messages=conversation_history.dump(), **kwargs
        )
        return ChainResponse(
            content=response.choices[0].message.content or "",
            metadata={
                "__tool_calls": self._prepare_tool_calls(response.choices[0].message.tool_calls)
            },
        )

    async def async_generate(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> str:
        conversation_history = conversation_history or ConversationHistory()
        response = await self._async_client.chat.completions.create(
            model=self.model_name, messages=conversation_history.dump(), **kwargs
        )
        return ChainResponse(
            content=response.choices[0].message.content or "",
            metadata={
                "__tool_calls": self._prepare_tool_calls(response.choices[0].message.tool_calls)
            },
        )

    def generate_stream(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> Generator[str, None, None]:
        conversation_history = conversation_history or ConversationHistory()

        for chunk in self._client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            stream=True,
            **kwargs,
        ):
            yield ChainResponse(
                content=chunk.choices[0].delta.content or "",
                metadata={
                    "__tool_calls": self._prepare_tool_calls(
                        chunk.choices[0].delta.tool_calls,
                    )
                },
            )

    async def async_generate_stream(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> Generator[str, None, None]:
        conversation_history = conversation_history or ConversationHistory()

        async for chunk in await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=conversation_history.dump(),
            stream=True,
            **kwargs,
        ):
            yield ChainResponse(
                content=chunk.choices[0].delta.content or "",
                metadata={
                    "__tool_calls": self._prepare_tool_calls(chunk.choices[0].delta.tool_calls)
                },
            )
