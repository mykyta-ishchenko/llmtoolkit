import asyncio
import os

from llmtoolkit.conversation import ConversationHistory, ConversationMessage
from llmtoolkit.llms import MistralAsyncModel


async def main():
    # Initialize the Ollama LLM model instance
    model_name = os.getenv("TOOLKIT__MISTRAL_MODEL_NAME", "ministral-8b-latest")
    mistral_api_key = os.getenv("TOOLKIT__MISTRAL_API_KEY")
    mistral_model = MistralAsyncModel(api_key=mistral_api_key, model_name=model_name)

    # Example prompt to the model
    prompt = "What's the latest news on climate change?"

    # Generate a response for a single prompt
    response = await mistral_model.generate(prompt)
    print(f"Generated response: {response}")

    # Prepare a conversation history object
    conversation_history = ConversationHistory()

    # Add an initial user message to the conversation history
    conversation_history.append(
        ConversationMessage(role="user", content="Tell me about AI advancements.")
    )

    # Generate a response with conversation history
    response_with_history = await mistral_model.generate(
        "What are the ethical implications?", conversation_history
    )
    print(f"Response with conversation history: {response_with_history}")

    # Streaming example
    print("Streaming response:")
    async for chunk in mistral_model.generate_stream(prompt):
        print(chunk, end="")  # Print each chunk as it arrives


if __name__ == "__main__":
    asyncio.run(main())
