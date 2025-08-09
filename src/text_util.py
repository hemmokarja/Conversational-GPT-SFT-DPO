PAD_TOKEN = "<|pad|>"


def add_pad_token_to_tokenizer(tokenizer, pad_token=PAD_TOKEN):
    tokenizer.add_special_tokens({"pad_token": pad_token})


def make_user_conversation(user_prompt):
    # create a conversation with a single user message
    return {
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }


def add_assistant_message(conversation, assistant_content):
    # add an assistant message to an existing conversation
    messages = conversation["messages"] + [
        {"role": "assistant", "content": assistant_content}
    ]
    return {"messages":  messages}


def get_last_assistant_message_content(conversation):
    # find assistant's last message and return its content
    for msg in reversed(conversation["messages"]):
        if msg["role"] == "assistant":
            return msg["content"]
    return
