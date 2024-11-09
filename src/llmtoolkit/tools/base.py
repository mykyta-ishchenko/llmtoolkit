from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .config import ToolDefinition


class Tool(ABC, BaseModel):
    definition: ToolDefinition

    @abstractmethod
    def call(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    async def async_call(self, *args, **kwargs) -> Any: ...
