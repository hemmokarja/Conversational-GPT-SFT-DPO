import re

import torch


_TURN_SEPARATOR = "!END"


class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __str__(self):
        return f"{self.role.capitalize()}: {self.content}"

    def __repr__(self):
        return f"Message(role={self.role!r}, content={self.content!r})"
    
    def to_dict(self):
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, msg_dict):
        return cls(msg_dict["role"], msg_dict["content"])


class Conversation:
    def __init__(self):
        self.messages = []

    def add_user_message(self, content):
        if self.messages and self.messages[-1].role == "user":
            raise ValueError("Cannot add user message when last message is from user")
        self.messages.append(Message("user", content))

    def add_assistant_message(self, content):
        if self.messages and self.messages[-1].role == "assistant":
            raise ValueError(
                "Cannot add assistant message when last message is from assistant"
            )
        self.messages.append(Message("assistant", content))

    def __str__(self):
        return "\n".join(str(m) for m in self.messages)

    def __repr__(self):
        return f"Conversation(messages={self.messages!r})"
    
    def to_dict(self):
        return {"messages": [msg.to_dict() for msg in self.messages]}

    @classmethod
    def from_dict(cls, conv_dict):
        conversation = cls()
        for msg_dict in conv_dict["messages"]:
            if msg_dict["role"] == "user":
                conversation.add_user_message(msg_dict["content"])
            elif msg_dict["role"] == "assistant":
                conversation.add_assistant_message(msg_dict["content"])
            else:
                raise ValueError(f"Unknown role '{msg_dict['role']}' in message dict")
        return conversation


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
        if isinstance(conversation, dict):
            conversation = Conversation.from_dict(conversation)
        if for_generation:
            return self.encode_conversation_for_generation_to_tokens(conversation)
        return self.encode_conversation_to_tokens(conversation)

    def encode_conversation_to_tokens(self, conversation):
        """
        Convert conversation to tokenized User/Assistant chat format for training.

        Inputs are assumed to be a Conversation object with Message objects.
        
        The conversations are then converted to chat format

            user: <content><endtoken>
            assistant: <content><endtoken>

        And the text is then tokenized.

        Args:
            conversation: Conversation object

        Returns:
            {"input_ids": [...], "labels": [...]}
        """
        self._validate_conversation(conversation)

        input_ids = []
        labels = []

        for msg in conversation.messages:
            msg_ids, msg_labels = self._encode_message(msg.role, msg.content)
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
            conversation: Conversation object

        Returns:
            {"input_ids": [...]}
        """
        self._validate_conversation(conversation)

        if len(conversation.messages) < 2:
            raise ValueError(
                f"Conversation must contain at least two messages (user, assistant), "
                f"got {len(conversation.messages)}"
            )

        if conversation.messages[-1].role != "assistant":
            raise ValueError(
                "Conversation must end with assistant message for generation"
            )

        input_ids = []
        messages = conversation.messages

        # process all messages except the last one normally
        for msg in messages[:-1]:
            role, content = msg.role, msg.content
            msg_ids, _ = self._encode_message(role, content)
            input_ids.extend(msg_ids)

        # process the final assistant message for generation
        role, content = messages[-1].role, messages[-1].content
        msg_ids = self._encode_message_for_generation(role, content)
        input_ids.extend(msg_ids)

        input_ids = input_ids[:self.max_length]
        return {"input_ids": input_ids}

    def decode_tokens_to_conversation(self, token_ids):
        """Decode token IDs back to Conversation object"""
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
        Parse User/Assistant formatted text back into Conversation object.
        Uses regex to extract role and content from each message block.
        """
        esc_end = re.escape(self.turn_separator)

        # pattern to match "role: content!END"
        pattern = rf"(user|assistant):\s*(.*?){esc_end}"
        matches = re.findall(pattern, chat_text, flags=re.DOTALL)

        conversation = Conversation()
        for role, content in matches:
            conversation.messages.append(Message(role.lower(), content.strip()))

        return conversation
    
    def _validate_conversation(self, conversation):
        """Validate Conversation object format."""
        if not isinstance(conversation, Conversation):
            raise ValueError("conversation must be a Conversation object")

        for i, msg in enumerate(conversation.messages):
            if not isinstance(msg, Message):
                raise ValueError(f"Message {i} must be a Message object")

            if msg.role not in ["user", "assistant"]:
                raise ValueError(f"Unknown role '{msg.role}' in message {i}")
