from llmtoolkit.conversation import ConversationHistory, ConversationMessage


def main():
    # Create a few conversation messages
    message1 = ConversationMessage(role="user", content="Hello, how are you?")
    message2 = ConversationMessage(
        role="assistant", content="I'm good, thank you! How can I assist you today?"
    )
    message3 = ConversationMessage(role="user", content="Can you tell me a joke?")
    message4 = ConversationMessage(
        role="assistant",
        content="Sure! Why don't scientists trust atoms? Because they make up everything!",
    )

    # Initialize the conversation history
    history = ConversationHistory(messages=[message1, message2])

    # Append a new message
    history.append(message3)

    # Extend the history with more messages
    history.extend([message4])

    # Access and print individual messages
    print(history[0].content)  # Output: "Hello, how are you?"
    print(history[1].content)  # Output: "I'm good, thank you! How can I assist you today?"

    # Iterate over all messages in history
    for message in history:
        print(f"{message.role}: {message.content}")

    # Output the length of the conversation history
    print(f"Total messages in history: {len(history)}")


if __name__ == "__main__":
    main()
