from typing import Any

from pydantic import BaseModel


class ToolProperty(BaseModel):
    type: str
    description: str

    class Config:
        extra = "allow"


class ToolParameters(BaseModel):
    type: str
    required: list[str]
    properties: dict[str, ToolProperty | dict[str, Any]]


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: ToolParameters


class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolFunction
