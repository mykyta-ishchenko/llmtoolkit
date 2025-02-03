from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

from llmtoolkit.core.enums import Roles

from .base import ResponseWithContext


class ConversationMessage(BaseModel):
    role: str
    content: str
    context: str | None = None

    class Config:
        extra = "allow"


class ConversationHistory(BaseModel):
    messages: list[ConversationMessage] = Field(default_factory=list)

    def __getitem__(self, index: int) -> ConversationMessage:
        return self.messages[index]

    def __setitem__(self, index: int, value: ConversationMessage) -> None:
        self.messages[index] = value

    def __iter__(self) -> Iterator[ConversationMessage]:
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def pop(self, index: int) -> ConversationMessage:
        return self.messages.pop(index)

    def extend(self, values: list[ConversationMessage]) -> None:
        self.messages.extend(values)

    def append(self, value: ConversationMessage) -> None:
        self.messages.append(value)

    def insert(self, index: int, value: ConversationMessage) -> None:
        self.messages.insert(index, value)

    def add_message(
        self,
        role: Roles,
        content: str,
        context: str | None = None,
    ) -> None:
        self.append(ConversationMessage(role=role, content=content, context=context))

    def add_user_message(
        self,
        content: str,
        context: str | None = None,
    ) -> None:
        self.add_message(
            Roles.USER,
            content,
            context,
        )

    def add_assistant_message(
        self,
        content: str,
        context: str | None = None,
    ) -> None:
        self.add_message(
            Roles.ASSISTANT,
            content,
            context,
        )

    def remove_system_message(self) -> ConversationMessage | None:
        if len(self) > 0 and self[0].role == Roles.SYSTEM:
            return self.pop(0)

    def set_system_message(self, content: str, context: str | None = None) -> None:
        self.remove_system_message()
        self.insert(0, ConversationMessage(role=Roles.SYSTEM, content=content, context=context))

    def dump(self) -> list[dict[str, Any]]:
        return self.model_dump().get("messages")


class ChainResponse(ResponseWithContext):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationParameters(BaseModel):
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_completion_tokens: int | None = None
    stop: list[str] | None = None
