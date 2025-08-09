import re

import torch


_TURN_SEPARATOR = "!END"


class ConversationPreprocessor:
    def __init__(
        self,
        tokenizer,
        ignored_idx=-100,
        max_length=None,
        turn_separator=_TURN_SEPARATOR,
    ):
        self.tokenizer = tokenizer
        self.ignored_idx = ignored_idx
        self.max_length = (
            tokenizer.model_max_length if max_length is None else max_length
        )
        self.turn_separator = turn_separator
        self.end_tokens = self.tokenizer.encode(turn_separator)

    def __call__(self, conversation, for_generation=False):
        if for_generation:
            return self.encode_conversation_for_generation_to_tokens(conversation)
        return self.encode_conversation_to_tokens(conversation)

    def encode_conversation_to_tokens(self, conversation):
        """
        Convert conversation to tokenized User/Assistant chat format for training.

        Inputs are assumed to conform to format

            {"messages": [{"role": "user|assistant", "content": "..."}]}
        
        The conversations are then converted to chat format

            user: <content><|endoftext|>
            assistant: <content><|endoftext|>

        And the text is then tokenized.

        Args:
            conversation: {"messages": [{"role": "user|assistant", "content": "..."}]}

        Returns:
            {"input_ids": [...], "labels": [...]}
        """
        self._validate_conversation(conversation)

        input_ids = []
        labels = []

        for msg in conversation["messages"]:
            role, content = msg["role"], msg["content"]

            msg_ids, msg_labels = self._encode_message(role, content)
            input_ids.extend(msg_ids)
            labels.extend(msg_labels)

        # shift labels by one position for next-token prediction
        labels = labels[1:] + [self.ignored_idx]

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        return {"input_ids": input_ids, "labels": labels}

    def encode_conversation_for_generation_to_tokens(self, conversation):
        """
        Convert conversation to tokenized chat format for generation.

        The conversation must end with an assistant message, which can have empty
        content ("", None).
        The final assistant message will be left incomplete (no eos_token) 
        to prompt the model to continue generation.

        Args:
            conversation: {"messages": [{"role": "user|assistant", "content": "..."}]}

        Returns:
            {"input_ids": [...]}
        """
        self._validate_conversation(conversation)

        if len(conversation["messages"]) < 2:
            raise ValueError(
                f"Conversation must contain at least two messages (user, assistant), "
                f"got {len(conversation['messages'])}"
            )

        if conversation["messages"][-1]["role"] != "assistant":
            raise ValueError(
                "Conversation must end with assistant message for generation"
            )

        input_ids = []
        messages = conversation["messages"]

        # process all messages except the last one normally
        for msg in messages[:-1]:
            role, content = msg["role"], msg["content"]
            msg_ids, _ = self._encode_message(role, content)
            input_ids.extend(msg_ids)

        # process the final assistant message for generation
        role, content = messages[-1]["role"], messages[-1]["content"]
        msg_ids = self._encode_message_for_generation(role, content)
        input_ids.extend(msg_ids)

        input_ids = input_ids[:self.max_length]
        return {"input_ids": input_ids}

    def decode_tokens_to_conversation(self, token_ids):
        """Decode token IDs back to conversation format"""
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                if token_ids.size(0) > 1:
                    raise ValueError("Batched decoding not supported")
                token_ids = token_ids[0]
            token_ids = token_ids.tolist()

        chat_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        conversation = self._parse_chat_text(chat_text)
        self._validate_conversation(conversation)
        return conversation

    def _encode_message(self, role, content):
        """Encode a single message into token IDs and labels."""
        ids = []
        labels = []

        # role + space
        role_ = role + ": "
        role_tokens = self.tokenizer.encode(role_, add_special_tokens=False)
        ids.extend(role_tokens)
        labels.extend([self.ignored_idx] * len(role_tokens))

        # content
        content_tokens = self.tokenizer.encode(
            content,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )
        ids.extend(content_tokens)

        if role == "assistant":
            labels.extend(content_tokens)
        else:
            labels.extend([self.ignored_idx] * len(content_tokens))

        # end
        end_tokens = self.tokenizer.encode(self.turn_separator, add_special_tokens=False)
        ids.extend(end_tokens)
        if role == "assistant":
            labels.extend(end_tokens)
        else:
            labels.extend([self.ignored_idx] * len(end_tokens))

        # \n
        ids.append(self.tokenizer.encode("\n", add_special_tokens=False)[0])
        labels.append(self.ignored_idx)

        return ids, labels

    def _encode_message_for_generation(self, role, content):
        """
        Encode a message for generation.

        This method is specifically for the final assistant message in generation,
        leaving it incomplete so the model continues from there.
        """
        if role != "assistant":
            raise ValueError("Generation encoding only supports assistant messages")

        ids = []

        # role + space
        role_ = role + ": "
        role_tokens = self.tokenizer.encode(role_, add_special_tokens=False)
        ids.extend(role_tokens)

        # content (if any)
        if content:
            content_tokens = self.tokenizer.encode(
                content,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length
            )
            ids.extend(content_tokens)

        return ids

    def _parse_chat_text(self, chat_text):
        """
        Parse User/Assistant formatted text back into conversation format.
        Uses regex to extract role and content from each message block.
        """
        esc_end = re.escape(self.turn_separator)

        # pattern to match "role: content!END"
        pattern = rf"(user|assistant):\s*(.*?){esc_end}"
        matches = re.findall(pattern, chat_text, flags=re.DOTALL)

        messages = []
        for role, content in matches:
            normalized_role = role.lower()
            messages.append({"role": normalized_role, "content": content.strip()})

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
