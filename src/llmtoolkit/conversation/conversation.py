from collections.abc import AsyncGenerator, Generator
from typing import Any

from pydantic import Field

from llmtoolkit.core import UNSET
from llmtoolkit.core.conversation import BaseConversation
from llmtoolkit.core.models import (
    ChainResponse,
    Context,
    GenerationParameters,
)


class Conversation(BaseConversation):
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)

    def _merge_generation_params(
        self,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        max_completion_tokens: int | None,
        stop: list[str] | None,
    ) -> dict[str, Any]:
        return dict(
            temperature=self.parameters.temperature if temperature is UNSET else temperature,
            top_p=self.parameters.top_p if top_p is UNSET else top_p,
            frequency_penalty=self.parameters.frequency_penalty
            if frequency_penalty is UNSET
            else frequency_penalty,
            presence_penalty=self.parameters.presence_penalty
            if presence_penalty is UNSET
            else presence_penalty,
            max_completion_tokens=self.parameters.max_completion_tokens
            if max_completion_tokens is UNSET
            else max_completion_tokens,
            stop=self.parameters.stop if stop is UNSET else stop,
        )

    def chat(
        self,
        prompt: str,
        context: Context | None = None,
        *,
        temperature: float = UNSET,
        top_p: float = UNSET,
        frequency_penalty: float = UNSET,
        presence_penalty: float = UNSET,
        max_completion_tokens: int | None = UNSET,
        stop: list[str] | None = UNSET,
        **kwargs,
    ) -> ChainResponse:
        self.history.add_user_message(prompt, context)
        generation_params = self._merge_generation_params(
            temperature, top_p, frequency_penalty, presence_penalty, max_completion_tokens, stop
        )
        response = self.chain.generate(
            conversation_history=self.history.copy(deep=True),
            **generation_params,
            **kwargs,
        )
        self.history.add_assistant_message(response.content)
        return response

    async def achat(
        self,
        prompt: str,
        context: Context | None = None,
        *,
        temperature: float = UNSET,
        top_p: float = UNSET,
        frequency_penalty: float = UNSET,
        presence_penalty: float = UNSET,
        max_completion_tokens: int | None = UNSET,
        stop: list[str] | None = UNSET,
        **kwargs,
    ) -> ChainResponse:
        self.history.add_user_message(prompt, context)
        generation_params = self._merge_generation_params(
            temperature, top_p, frequency_penalty, presence_penalty, max_completion_tokens, stop
        )
        response = await self.chain.agenerate(
            conversation_history=self.history.copy(deep=True),
            **generation_params,
            **kwargs,
        )
        self.history.add_assistant_message(response.content)
        return response

    def stream(
        self,
        prompt: str,
        context: Context | None = None,
        *,
        temperature: float = UNSET,
        top_p: float = UNSET,
        frequency_penalty: float = UNSET,
        presence_penalty: float = UNSET,
        max_completion_tokens: int | None = UNSET,
        stop: list[str] | None = UNSET,
        **kwargs,
    ) -> Generator[ChainResponse, None, None]:
        self.history.add_user_message(prompt, context)
        history = self.history.copy(deep=True)
        self.history.add_user_message("")
        generation_params = self._merge_generation_params(
            temperature, top_p, frequency_penalty, presence_penalty, max_completion_tokens, stop
        )
        for chunk in self.chain.stream(
            conversation_history=history,
            **generation_params,
            **kwargs,
        ):
            self.history[-1].content += chunk.content
            yield chunk
        self.history.add_assistant_message("")

    async def astream(
        self,
        prompt: str,
        context: Context | None = None,
        *,
        temperature: float = UNSET,
        top_p: float = UNSET,
        frequency_penalty: float = UNSET,
        presence_penalty: float = UNSET,
        max_completion_tokens: int | None = UNSET,
        stop: list[str] | None = UNSET,
        **kwargs,
    ) -> AsyncGenerator[ChainResponse, None]:
        self.history.add_user_message(prompt, context)
        history = self.history.copy(deep=True)
        self.history.add_assistant_message("")
        generation_params = self._merge_generation_params(
            temperature, top_p, frequency_penalty, presence_penalty, max_completion_tokens, stop
        )
        async for chunk in self.chain.astream(
            conversation_history=history,
            **generation_params,
            **kwargs,
        ):
            self.history[-1].content += chunk.content
            yield chunk
