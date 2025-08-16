PAD_TOKEN = "<|pad|>"


def add_pad_token_to_tokenizer(tokenizer, pad_token=PAD_TOKEN):
    tokenizer.add_special_tokens({"pad_token": pad_token})
