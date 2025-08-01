def preprocess_fn(conversation, tokenizer, ignored_idx=-100, max_length=None):
    """
    1. Make conversations conform to ChatML format
    
        <|im_start|>user
        How do I bake a cake?<|im_end|>
        <|im_start|>assistant
        Here's how to bake a basic cake...<|im_end|>

    2. Tokenize
    3. Label tokens, ensuring that loss is computed only across assistant message 
       content tokens
    4. Shift by one and truncate

    Expects input conversation to conform to format

        {
            "messages": [
                {"role": "user", "content": "How do I bake a cake?"},
                {"role": "assistant", "content": "Here's how to bake a basic cake..."}
            ]
        }
    """
    max_length = tokenizer.model_max_length if max_length is None else max_length

    input_ids = []
    labels = []

    for msg in conversation["messages"]:
        role = msg["role"]

        if role not in ["user", "assistant"]:
            raise ValueError(f"Unknown role '{role}', expected 'user' or 'assistant'")


        content = msg["content"]

        # im start
        input_ids.append(tokenizer.im_start_token_id)
        labels.append(ignored_idx)  # don't compute loss on im start token

        # role
        role_tokens = tokenizer.encode(role, add_special_tokens=False)
        input_ids.extend(role_tokens)
        labels.extend([ignored_idx] * len(role_tokens))  # don't compute loss on role

        # new line
        newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]
        input_ids.append(newline_token)
        labels.append(ignored_idx)

        # msg content (set msg content labels based on role)
        content_tokens = tokenizer.encode(
            content, add_special_tokens=False, truncation=True, max_length=max_length
        )
        input_ids.extend(content_tokens)
        if role == "assistant":
            labels.extend(content_tokens)  # compute loss on assistant content
        else:
            labels.extend([ignored_idx] * len(content_tokens))  # don't compute loss on user content

        # im end
        input_ids.append(tokenizer.im_end_token_id)
        if role == "assistant":
            labels.append(tokenizer.im_end_token_id)
        else:
            labels.append(ignored_idx)

        # new line
        input_ids.append(newline_token)
        labels.append(ignored_idx)

    # shift and right pad labels
    labels = labels[1:] + [ignored_idx]

    # truncate if necessary
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "labels": labels}
