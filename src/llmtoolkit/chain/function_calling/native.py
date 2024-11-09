import json
from collections.abc import Generator

from pydantic import Field

from llmtoolkit.chain.base import Chain
from llmtoolkit.core.models import (
    ChainResponse,
    ConversationHistory,
    ConversationMessage,
    Roles,
    ToolCall,
)
from llmtoolkit.exc import FunctionCallingError
from llmtoolkit.tools import Tool


class NativeFunctionCallingChain(Chain):
    chain: Chain
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: str = "auto"

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
            try:
                res = tools_dict[tool_name].call(**tool_args)
            except Exception as exc:
                res = f"{str(FunctionCallingError())}: {str(exc)}"
            conversation_history.append(
                ConversationMessage(
                    role=Roles.TOOL,
                    content=json.dumps(res),
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
            )

    @staticmethod
    async def _async_update_conversation_with_tool_calls(
        conversation_history: ConversationHistory, tool_calls: list, tools_dict: dict
    ) -> None:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            try:
                res = await tools_dict[tool_name].async_call(**tool_args)
            except Exception as exc:
                res = f"{str(FunctionCallingError())}: {str(exc)}"
            conversation_history.append(
                ConversationMessage(
                    role=Roles.TOOL,
                    content=json.dumps(res),
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
            )

    @staticmethod
    def _combine_chunks(chunks: list[ToolCall]) -> list[ToolCall]:
        combined = []
        for chunk in chunks:
            if chunk.id:
                combined.append(chunk)
            else:
                combined[chunk.index].function.arguments += chunk.function.arguments
        return combined

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
                tool_choice=self.tool_choice,
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
                tool_choice=self.tool_choice,
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

    @staticmethod
    def _process_chunk(
        chunk: ChainResponse,
        streamed_chunks: list[ChainResponse],
        tool_calls: list[ToolCall],
        used_tool_calls: list[ToolCall],
    ) -> list[ToolCall]:
        new_tool_calls = chunk.metadata.get("__tool_calls") or []
        for new_tool_call in new_tool_calls:
            if new_tool_call.index is not None:
                streamed_chunks.append(new_tool_call)
            else:
                tool_calls.append(new_tool_call)
        chunk.metadata["used_tool_calls"] = used_tool_calls

    def generate_stream(
        self,
        conversation_history: ConversationHistory | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        used_tool_calls = []
        tools_dict = self._tools_dict(tools)
        tools_list = [tool.definition.model_dump() for tool in tools_dict.values()]

        while True:
            response = self.chain.generate_stream(
                conversation_history,
                tools=tools_list,
                tool_choice=self.tool_choice,
                **kwargs,
            )

            tool_calls = []
            streamed_chunks = []

            for chunk in response:
                self._process_chunk(chunk, streamed_chunks, tool_calls, used_tool_calls)
                yield chunk

            combined_chunks = self._combine_chunks(streamed_chunks)
            tool_calls.extend(combined_chunks)

            if len(tool_calls) == 0:
                return

            used_tool_calls.extend(tool_calls)
            conversation_history.append(
                ConversationMessage(
                    role=Roles.ASSISTANT,
                    content="",
                    tool_calls=tool_calls,
                )
            )

            self._update_conversation_with_tool_calls(conversation_history, tool_calls, tools_dict)

    async def async_generate_stream(
        self,
        conversation_history: ConversationHistory | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChainResponse, None, None]:
        conversation_history = conversation_history or ConversationHistory()
        used_tool_calls = []
        tools_dict = self._tools_dict(tools)
        tools_list = [tool.definition.model_dump() for tool in tools_dict.values()]

        while True:
            response = self.chain.async_generate_stream(
                conversation_history,
                tools=tools_list,
                tool_choice=self.tool_choice,
                **kwargs,
            )

            tool_calls = []
            streamed_chunks = []

            async for chunk in response:
                self._process_chunk(chunk, streamed_chunks, tool_calls, used_tool_calls)
                yield chunk

            combined_chunks = self._combine_chunks(streamed_chunks)
            tool_calls.extend(combined_chunks)

            if len(tool_calls) == 0:
                return

            used_tool_calls.extend(tool_calls)
            conversation_history.append(
                ConversationMessage(
                    role=Roles.ASSISTANT,
                    content="",
                    tool_calls=tool_calls,
                )
            )

            await self._async_update_conversation_with_tool_calls(
                conversation_history, tool_calls, tools_dict
            )
