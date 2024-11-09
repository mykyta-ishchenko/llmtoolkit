from abc import ABC

from llmtoolkit.chain import Chain


class BaseLLM(Chain, ABC):
    model_name: str = "default"
