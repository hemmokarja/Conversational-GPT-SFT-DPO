class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __str__(self):
        return f"{self.role.capitalize()}: {self.content}"

    def __repr__(self):
        return f"Message(role={self.role!r}, content={self.content!r})"

    def __eq__(self, other):
        if not isinstance(other, Message):
            return NotImplemented
        return self.role == other.role and self.content == other.content

    def __ne__(self, other):
        return not self.__eq__(other)

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
        if content is None:
            raise ValueError("Cannot add None for user message content")
        self.messages.append(Message("user", content))

    def add_assistant_message(self, content):
        if self.messages and self.messages[-1].role == "assistant":
            raise ValueError(
                "Cannot add assistant message when last message is from assistant"
            )
        self.messages.append(Message("assistant", content))

    def delete_last_message(self):
        return self.messages.pop(-1)

    def __str__(self):
        return "\n".join(str(m) for m in self.messages)

    def __repr__(self):
        return f"Conversation(messages={self.messages!r})"

    def __getitem__(self, index):
        return self.messages[index]

    def __len__(self):
        return len(self.messages)

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
