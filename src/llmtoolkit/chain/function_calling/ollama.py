import json
from collections.abc import Generator

from pydantic import Field

from llmtoolkit.chain.base import Chain
from llmtoolkit.core.models import (
    ChainResponse,
    ConversationHistory,
    ConversationMessage,
    Roles,
)
from llmtoolkit.exc import NativeFunctionCallingSupportError
from llmtoolkit.tools import Tool


class OllamaFunctionCallingChain(Chain):
    chain: Chain
    tools: list[Tool] = Field(default_factory=list)

    def _tools_dict(self, tools: list[Tool] | None = None) -> dict[str, Tool]:
        return {
            tool.definition.function.name: tool for tool in self.tools + (tools if tools else [])
        }

    @staticmethod
    def _update_conversation_with_tool_calls(
        conversation_history: ConversationHistory, tool_calls: list, tools_dict: dict
    ) -> None:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            res = (
                tools_dict[tool_name].call(**tool_args)
                if not callable(tools_dict[tool_name].async_call)
                else None
            )
            conversation_history.append(
                ConversationMessage(
                    role=Roles.TOOL,
                    content=json.dumps(res),
                    name=tool_name,
                )
            )

    @staticmethod
    async def _async_update_conversation_with_tool_calls(
        conversation_history: ConversationHistory, tool_calls: list, tools_dict: dict
    ) -> None:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            res = await tools_dict[tool_name].async_call(**tool_args)
            conversation_history.append(
                ConversationMessage(
                    role=Roles.TOOL,
                    content=json.dumps(res),
                    name=tool_name,
                )
            )

    def generate(
        self,
        conversation_history: ConversationHistory | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> ChainResponse:
        conversation_history = conversation_history or ConversationHistory()
        used_tool_calls = []
        tools_dict = self._tools_dict(tools)
        tools_list = [tool.definition.model_dump() for tool in tools_dict.values()]

        while True:
            response = self.chain.generate(
                conversation_history,
                tools=tools_list,
                **kwargs,
            )
            tool_calls = response.metadata.get("__tool_calls")

            if not tool_calls:
                break

            used_tool_calls += tool_calls
            conversation_history.append(
                ConversationMessage(
                    role=Roles.ASSISTANT,
                    content=response.content,
                    tool_calls=tool_calls,
                )
            )

            self._update_conversation_with_tool_calls(conversation_history, tool_calls, tools_dict)

        response.metadata["used_tool_calls"] = used_tool_calls
        return response

    async def async_generate(
        self,
        conversation_history: ConversationHistory | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> ChainResponse:
        conversation_history = conversation_history or ConversationHistory()
        used_tool_calls = []
        tools_dict = self._tools_dict(tools)
        tools_list = [tool.definition.model_dump() for tool in tools_dict.values()]

        while True:
            response = await self.chain.async_generate(
                conversation_history,
                tools=tools_list,
                **kwargs,
            )
            tool_calls = response.metadata.get("__tool_calls")

            if not tool_calls:
                break

            used_tool_calls += tool_calls
            conversation_history.append(
                ConversationMessage(
                    role=Roles.ASSISTANT,
                    content=response.content,
                    tool_calls=tool_calls,
                )
            )

            await self._async_update_conversation_with_tool_calls(
                conversation_history, tool_calls, tools_dict
            )

        response.metadata["used_tool_calls"] = used_tool_calls
        return response

    def generate_stream(
        self,
        conversation_history: ConversationHistory | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChainResponse, None, None]:
        raise NativeFunctionCallingSupportError

    async def async_generate_stream(
        self,
        conversation_history: ConversationHistory | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChainResponse, None, None]:
        raise NativeFunctionCallingSupportError
