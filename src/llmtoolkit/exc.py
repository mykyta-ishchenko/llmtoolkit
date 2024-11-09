class BaseLLMToolkitException(Exception):
    message: str = "An error occurred in the LLM-Toolkit library."

    def __init__(self, message: str = None):
        super().__init__(message or self.message)


class StreamReadError(BaseLLMToolkitException):
    message: str = "An error occurred while reading stream."


class NativeFunctionCallingSupportError(BaseLLMToolkitException):
    message: str = "Native function calling is not supported for this model or run mode."


class FunctionCallingError(BaseLLMToolkitException):
    message = str = "Error occurred while calling function."
