from typing import Any

from pydantic import BaseModel, Field


class Context(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)


class ResponseWithContext(BaseModel):
    context: Context = Field(default_factory=Context)
