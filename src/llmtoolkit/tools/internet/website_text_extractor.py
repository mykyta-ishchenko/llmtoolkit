from typing import Any

import httpx
from bs4 import BeautifulSoup
from pydantic import Field

from llmtoolkit.tools.config import (
    ToolDefinition,
    ToolFunction,
    ToolParameters,
    ToolProperty,
)

from .base import BaseInternetTool


class WebsiteTextExtractor(BaseInternetTool):
    client: httpx.Client | None = None
    async_client: httpx.AsyncClient | None = None

    definition: ToolDefinition = Field(
        default_factory=lambda: ToolDefinition(
            function=ToolFunction(
                name="WebsiteTextExtractor",
                description="Extract the main text content from a specific web page given its URL.",
                parameters=ToolParameters(
                    type="object",
                    properties=dict(
                        url=ToolProperty(
                            type="string",
                            description="URL of the website to extract text from.",
                        ),
                    ),
                    required=["url"],
                ),
            )
        )
    )

    def model_post_init(self, __context: Any) -> None:
        self.client = self.client or httpx.Client()
        self.async_client = self.async_client or httpx.AsyncClient()

    @staticmethod
    def _get_text_from_html(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    def call(self, url: str) -> str:
        response = self.client.get(url)
        response.raise_for_status()
        return self._get_text_from_html(response.text)

    async def async_call(self, url: str) -> str:
        response = await self.async_client.get(url)
        response.raise_for_status()
        return self._get_text_from_html(response.text)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.client:
            self.client.close()
        if self.async_client:
            await self.async_client.aclose()

    class Config:
        arbitrary_types_allowed = True
        model_dump_exclude = {"client", "async_client"}
