import asyncio

from llmtoolkit.conversation import Conversation
from llmtoolkit.llms import OllamaLLMModel


async def main():
    llm = OllamaLLMModel(model_name="llama3.1:8b", host="http://localhost:11434")

    conversation = Conversation(llm=llm)

    # Start a chat interaction
    user_input = "What's the weather like today?"
    response = await conversation.chat(user_input)
    print(f"Assistant: {response}")

    # Stream a long response (for example, a story or detailed explanation)
    user_input = "Can you tell me a story?"
    async for part in conversation.stream(user_input):
        print(f"Assistant (streamed): {part}", end="")


if __name__ == "__main__":
    asyncio.run(main())
