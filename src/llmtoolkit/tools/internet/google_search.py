from typing import Any

import httpx
from pydantic import Field, SecretStr

from llmtoolkit.tools.config import (
    ToolDefinition,
    ToolFunction,
    ToolParameters,
    ToolProperty,
)

from .base import BaseInternetTool


class GoogleAPISearch(BaseInternetTool):
    api_key: SecretStr
    cx_id: SecretStr

    service_url: str = "https://www.googleapis.com/customsearch/v1"

    client: httpx.Client | None = None
    async_client: httpx.AsyncClient | None = None

    definition: ToolDefinition = Field(
        default_factory=lambda: ToolDefinition(
            function=ToolFunction(
                name="GoogleAPISearch",
                description="Perform a web search using Google API and retrieve relevant search "
                "results. Use it only if you can't answer without information from "
                "the Internet.",
                parameters=ToolParameters(
                    type="object",
                    properties=dict(
                        query=ToolProperty(
                            type="string",
                            description="Search query for Google.",
                        ),
                    ),
                    required=["query"],
                ),
            )
        )
    )

    def model_post_init(self, __context: Any) -> None:
        self.client = self.client or httpx.Client()
        self.async_client = self.async_client or httpx.AsyncClient()

    def _get_params(self, query: str) -> dict[str, str]:
        return {
            "q": query,
            "cx": self.cx_id.get_secret_value(),
            "key": self.api_key.get_secret_value(),
        }

    @staticmethod
    def _data_from_response(response: httpx.Response) -> dict[str, str]:
        response.raise_for_status()
        return [
            {
                "title": result.get("title"),
                "url": result.get("link"),
                "snippet": result.get("snippet"),
            }
            for result in response.json().get("items", [])
        ]

    def call(self, query: str) -> list[dict[str, str]]:
        response = self.client.get(self.service_url, params=self._get_params(query))
        return self._data_from_response(response)

    async def async_call(self, query: str) -> list[dict[str, str]]:
        response = await self.async_client.get(self.service_url, params=self._get_params(query))
        return self._data_from_response(response)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.client:
            self.client.close()
        if self.async_client:
            await self.async_client.aclose()

    class Config:
        arbitrary_types_allowed = True
        model_dump_exclude = {"client", "async_client"}
