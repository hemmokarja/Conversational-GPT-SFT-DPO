import re

import torch

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


class ChatMLPreprocessor:
    def __init__(self, tokenizer, ignored_idx=-100, max_length=None):
        self.tokenizer = tokenizer
        self.ignored_idx = ignored_idx
        self.max_length = (
            tokenizer.model_max_length if max_length is None else max_length
        )

    def __call__(self, conversation, return_labels=True):
        return self.encode_conversation_to_tokens(conversation, return_labels)

    def encode_conversation_to_tokens(self, conversation, return_labels):
        """
        Convert conversation to tokenized ChatML format for training.

        Args:
            conversation: {"messages": [{"role": "user|assistant", "content": "..."}]}
            return_labels: Whether to return labels for loss computation
            
        Returns:
            {"input_ids": [...], "labels": [...]} (labels only if return_labels=True)
        """
        self._validate_conversation(conversation)

        input_ids = []
        labels = []
        
        for msg in conversation["messages"]:
            role, content = msg["role"], msg["content"]
            
            # <|im_start|>role\ncontent<|im_end|>\n
            msg_ids, msg_labels = self._encode_message(role, content)
            input_ids.extend(msg_ids)
            labels.extend(msg_labels)

        input_ids = input_ids[:self.max_length]

        result = {"input_ids": input_ids}

        if return_labels:
            # shift labels by one position for next-token prediction
            labels = labels[1:] + [self.ignored_idx]
            labels = labels[:self.max_length]
            result["labels"] = labels

        return result

    def decode_tokens_to_conversation(self, token_ids):
        """Decode token IDs back to conversation format"""
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                if token_ids.size(0) > 1:
                    raise ValueError("Batched decoding not supported")
                token_ids = token_ids[0]
            token_ids = token_ids.tolist()

        chatml_text = decode_by_token(
            self.tokenizer, token_ids, skip_special_tokens=False
        )
        conversation = self._parse_chatml_text(chatml_text)
        self._validate_conversation(conversation)
        return conversation

    def _encode_message(self, role, content):
        """Encode a single message into token IDs and labels."""
        ids = []
        labels = []

        # <|im_start|>
        ids.append(self.tokenizer.im_start_token_id)
        labels.append(self.ignored_idx)

        # role
        role_tokens = self.tokenizer.encode(role, add_special_tokens=False)
        ids.extend(role_tokens)
        labels.extend([self.ignored_idx] * len(role_tokens))

        # \n
        ids.append(self.tokenizer.encode("\n", add_special_tokens=False)[0])
        labels.append(self.ignored_idx)

        # content
        content_tokens = self.tokenizer.encode(
            content, add_special_tokens=False, truncation=True, max_length=self.max_length
        )
        ids.extend(content_tokens)

        if role == "assistant":
            labels.extend(content_tokens)
        else:
            labels.extend([self.ignored_idx] * len(content_tokens))

        # <|im_end|>
        ids.append(self.tokenizer.im_end_token_id)
        if role == "assistant":
            labels.append(self.tokenizer.im_end_token_id)
        else:
            labels.append(self.ignored_idx)

        # \n
        ids.append(self.tokenizer.encode("\n", add_special_tokens=False)[0])
        labels.append(self.ignored_idx)

        return ids, labels
    
    def _parse_chatml_text(self, chatml_text):
        """
        Parse ChatML formatted text back into conversation format.

        Uses regex to robustly extract role and content from each message block.
        """
        esc_start = re.escape(self.tokenizer.im_start_token)
        esc_end = re.escape(self.tokenizer.im_end_token)

        pattern = rf"{esc_start}\s*(\w+)\n(.*?){esc_end}"
        matches = re.findall(pattern, chatml_text, flags=re.DOTALL)

        messages = []
        for role, content in matches:
            messages.append({"role": role, "content": content.strip()})

        return {"messages": messages}
    
    def _validate_conversation(self, conversation):
        """Validate conversation format."""
        if "messages" not in conversation:
            raise ValueError("Conversation must have 'messages' key")

        for i, msg in enumerate(conversation["messages"]):
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message {i} must have 'role' and 'content' keys")

            if msg["role"] not in ["user", "assistant"]:
                raise ValueError(f"Unknown role '{msg['role']}' in message {i}")
