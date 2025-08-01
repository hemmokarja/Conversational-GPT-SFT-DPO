def decode_by_token(tokenizer, ids):
    # avoid spaces around ChatML im tokens, can't get rid of them when applying calling
    # tokenizer.decode(ids) directly
    return "".join([tokenizer.decode(i) for i in ids])
