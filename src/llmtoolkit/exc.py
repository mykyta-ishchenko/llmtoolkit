class BaseLLMToolkitException(Exception):
    message: str = "An error occurred in the LLM-Toolkit library."

    def __init__(self, message: str = None):
        super().__init__(message or self.message)


class StreamReadError(BaseLLMToolkitException):
    message: str = "An error occurred while reading stream."


class NotImplementedToolkitError(BaseLLMToolkitException):
    message: str = "Not implemented."


class UnsupportedFormatError(BaseLLMToolkitException):
    message: str = "Unsupported format."


class FfmpegError(BaseLLMToolkitException):
    message: str = "Ffmpeg installation required."
