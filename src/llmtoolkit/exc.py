"""
Module for defining custom exceptions in the LLM-Toolkit library.

This module provides a base exception class, `BaseLLMToolkitException`, for handling general
errors within the library, as well as specific exceptions for more granular error handling.
"""


class BaseLLMToolkitException(Exception):
    """
    Base exception class for errors in the LLM-Toolkit library.

    Attributes:
        message (str): A description of the error.
    """

    message: str = "An error occurred in the LLM-Toolkit library."

    def __init__(self, message: str = None):
        super().__init__(message or self.message)


class OllamaConnectionError(BaseLLMToolkitException):
    """
    Exception raised when a connection to the Ollama host cannot be established.

    Attributes:
        message (str): Error message indicating connection failure.
    """

    message: str = "Can't connect to Ollama host."
