from llmtoolkit.conversation import Conversation
from llmtoolkit.llms import OllamaModel


def main():
    llm = OllamaModel(model_name="llama3.1:8b", host="http://localhost:11434")

    conversation = Conversation(llm=llm)
    print(conversation.history)

    # Start a chat interaction
    user_input = "What's the weather like today?"
    response = conversation.chat(user_input)
    print(f"Assistant: {response}")

    # Stream a long response (for example, a story or detailed explanation)
    user_input = "Can you tell me a story?"
    print("Assistant (streamed):", end="")
    for part in conversation.stream(user_input):
        print(part, end="")


if __name__ == "__main__":
    main()
