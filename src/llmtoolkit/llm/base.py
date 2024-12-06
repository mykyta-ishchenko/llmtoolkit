from abc import ABC

from pydantic import SecretStr

from llmtoolkit.chain import Chain


class BaseLLM(Chain, ABC):
    api_key: SecretStr = ""
    host: str | None = None
    model_name: str = "default"
