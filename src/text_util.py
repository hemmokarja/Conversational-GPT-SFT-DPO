PAD_TOKEN = "<|pad|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"


def add_pad_token_to_tokenizer(tokenizer, pad_token=PAD_TOKEN):
    tokenizer.add_special_tokens({"pad_token": pad_token})


def add_chatml_tokens_to_tokenizer(
    tokenizer, im_start_token=IM_START_TOKEN, im_end_token=IM_END_TOKEN
):
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [im_start_token, im_end_token]}
    )

    tokenizer.im_start_token = im_start_token
    tokenizer.im_end_token = im_end_token

    tokenizer.im_start_token_id = tokenizer.convert_tokens_to_ids(im_start_token)
    tokenizer.im_end_token_id = tokenizer.convert_tokens_to_ids(im_end_token)


def decode_by_token(tokenizer, ids, **tokenizer_kwargs):
    # avoid spaces around ChatML im tokens, can't get rid of them when applying calling
    # tokenizer.decode(ids) directly
    return "".join([tokenizer.decode(i, **tokenizer_kwargs) for i in ids])


def make_user_conversation(user_prompt):
    # create a conversation with a single user message
    return {
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }


def get_last_assistant_message_content(conversation):
    # find assistant's last message and return its content
    for msg in reversed(conversation["messages"]):
        if msg["role"] == "assistant":
            return msg["content"]
    return


def add_assistant_message(conversation, assistant_content):
    # TODO is this needed?
    # add an assistant message to an existing conversation
    messages = conversation["messages"] + [
        {"role": "assistant", "content": assistant_content}
    ]
    return {"messages":  messages}
