"""
Module for defining custom exceptions for the LLM-Toolkit library.
"""


class BaseLLMToolkitException(Exception):
    """
    Base exception class for the LLM-Toolkit library.

    Attributes:
        message (str): The error message associated with the exception.
    """

    message: str = "Base exception"

    def __init__(self):
        """
        Initializes the BaseLLMToolkitException instance with a default message.
        """
        super().__init__(self.message)


class OllamaConnectionError(BaseLLMToolkitException):
    message: str = "Can't connect to Ollama host"
