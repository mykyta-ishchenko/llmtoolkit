from typing import Any

from duckduckgo_search import DDGS, AsyncDDGS
from pydantic import Field, PrivateAttr

from llmtoolkit.tools.config import (
    ToolDefinition,
    ToolFunction,
    ToolParameters,
    ToolProperty,
)

from .base import BaseInternetTool


class DuckDuckGoSearch(BaseInternetTool):
    proxy: str | None = None

    _client: DDGS = PrivateAttr()
    _async_client: AsyncDDGS = PrivateAttr()

    definition: ToolDefinition = Field(
        default_factory=lambda: ToolDefinition(
            function=ToolFunction(
                name="DuckDuckGoSearch",
                description="Perform a web search using DuckDuckGo API and retrieve relevant search"
                " results. Use it only if you can't answer without information from "
                "the Internet.",
                parameters=ToolParameters(
                    type="object",
                    properties=dict(
                        query=ToolProperty(
                            type="string",
                            description="Search query for DuckDuckGo.",
                        ),
                    ),
                    required=["query"],
                ),
            )
        )
    )

    def model_post_init(self, __context: Any) -> None:
        self._client = DDGS(proxy=self.proxy)
        self._async_client = AsyncDDGS(proxy=self.proxy)

    @staticmethod
    def _data_from_response(results: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "title": result.get("title"),
                "url": result.get("href"),
                "snippet": result.get("body"),
            }
            for result in results
        ]

    def call(self, query: str) -> list[dict[str, str]]:
        results = self._client.text(query)
        return self._data_from_response(results)

    async def async_call(self, query: str) -> list[dict[str, str]]:
        results = await self._async_client.atext(query)
        return self._data_from_response(results)

    class Config:
        arbitrary_types_allowed = True
        model_dump_exclude = {"client", "async_client"}
