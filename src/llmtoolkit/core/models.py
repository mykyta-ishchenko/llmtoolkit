from collections.abc import Iterator
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationMessage(BaseModel):
    role: str
    content: str
    context: str | None = None

    class Config:
        extra = "allow"


class ConversationHistory(BaseModel):
    messages: list[ConversationMessage] = Field(default_factory=list)

    def __getitem__(self, key: int) -> ConversationMessage:
        return self.messages[key]

    def __setitem__(self, key: int, value: ConversationMessage) -> None:
        self.messages[key] = value

    def __iter__(self) -> Iterator[ConversationMessage]:
        return iter(self.messages)

    def extend(self, values: list[ConversationMessage]) -> None:
        self.messages.extend(values)

    def append(self, value: ConversationMessage) -> None:
        self.messages.append(value)

    def insert(self, index: int, value: ConversationMessage) -> None:
        self.messages.insert(index, value)

    def __len__(self) -> int:
        return len(self.messages)

    def dump(self) -> list[dict[str, Any]]:
        return self.model_dump().get("messages")


class ChainResponse(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallFunction(BaseModel):
    name: str | None
    arguments: str | None


class ToolCall(BaseModel):
    id: str | None = None
    index: int | None = None
    type: str | None = "function"
    function: ToolCallFunction = Field(default_factory=ToolCallFunction)
