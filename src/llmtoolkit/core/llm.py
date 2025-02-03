from abc import ABC

from .chain import Chain


class BaseLLM(Chain, ABC):
    api_key: str = "-"
    host: str | None = None
    model_name: str = "default"
