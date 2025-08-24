import re
from abc import ABC, abstractmethod

import torch
from transformers import logging

from src.conversation import Conversation, Message

logging.set_verbosity_error()

_TURN_SEPARATOR = "!END"


class BaseConversationPreprocessor(ABC):
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
    
    def __call__(self, conversation):
        if not isinstance(conversation, Conversation):
            # handle both dict and LazyRow objects
            if hasattr(conversation, "keys") or hasattr(conversation, "__getitem__"):
                conv_dict = dict(conversation)
                conversation = Conversation.from_dict(conv_dict)
            else:
                raise ValueError(f"Cannot convert {type(conversation)} to Conversation")
        return self.encode_to_tokens(conversation)

    def _encode_message(self, message):
        """Encode a single message into token IDs and labels."""
        ids = []
        labels = []

        # role + space
        role_ = message.role + ": "
        role_tokens = self.tokenizer.encode(role_, add_special_tokens=False)
        ids.extend(role_tokens)
        labels.extend([self.ignored_idx] * len(role_tokens))

        # content
        content_tokens = self.tokenizer.encode(
            message.content, add_special_tokens=False,
        )
        ids.extend(content_tokens)

        if message.role == "assistant":
            labels.extend(content_tokens)
        else:
            labels.extend([self.ignored_idx] * len(content_tokens))

        # end
        end_tokens = self.tokenizer.encode(self.turn_separator, add_special_tokens=False)
        ids.extend(end_tokens)
        if message.role == "assistant":
            labels.extend(end_tokens)
        else:
            labels.extend([self.ignored_idx] * len(end_tokens))

        # \n
        ids.append(self.tokenizer.encode("\n", add_special_tokens=False)[0])
        labels.append(self.ignored_idx)

        return ids, labels

    def _validate_conversation(self, conversation):
        """Validate Conversation object format."""
        if not isinstance(conversation, Conversation):
            raise ValueError(f"conversation must be a Conversation object, got {conversation}")

        for i, msg in enumerate(conversation.messages):
            if not isinstance(msg, Message):
                raise ValueError(f"Message {i} must be a Message object")

            if msg.role not in ["user", "assistant"]:
                raise ValueError(f"Unknown role '{msg.role}' in message {i}")

    @abstractmethod
    def encode_to_tokens(self, conversation):
        pass


class SFTConversationPreprocessor(BaseConversationPreprocessor):
    def __init__(
        self,
        tokenizer,
        ignored_idx=-100,
        max_length=None,
        turn_separator=_TURN_SEPARATOR,
    ):
        super().__init__(tokenizer, ignored_idx, max_length, turn_separator)

    def encode_to_tokens(self, conversation):
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

        for message in conversation.messages:
            msg_ids, msg_labels = self._encode_message(message)
            input_ids.extend(msg_ids)
            labels.extend(msg_labels)

        # shift labels by one position for next-token prediction
        labels = labels[1:] + [self.ignored_idx]

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        return {"input_ids": input_ids, "labels": labels}


class GenerationConversationPreprocessor(BaseConversationPreprocessor):
    def __init__(
        self,
        tokenizer,
        ignored_idx=-100,
        max_length=None,
        turn_separator=_TURN_SEPARATOR,
    ):
        super().__init__(tokenizer, ignored_idx, max_length, turn_separator)

    def encode_to_tokens(self, conversation):
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
        for message in messages[:-1]:
            msg_ids, _ = self._encode_message(message)
            input_ids.extend(msg_ids)

        # process the final assistant message for generation
        msg_ids = self._encode_message_for_generation(messages[-1])
        input_ids.extend(msg_ids)

        input_ids = input_ids[:self.max_length]
        return {"input_ids": input_ids}

    def _encode_message_for_generation(self, message):
        """
        Encode a message for generation.

        This method is specifically for the final assistant message in generation,
        leaving it incomplete so the model continues from there.
        """
        if message.role != "assistant":
            raise ValueError("Generation encoding only supports assistant messages")

        ids = []

        # role + space
        role_ = message.role + ": "
        role_tokens = self.tokenizer.encode(role_, add_special_tokens=False)
        ids.extend(role_tokens)

        # content (if any)
        if message.content:
            content_tokens = self.tokenizer.encode(
                message.content, add_special_tokens=False,
            )
            ids.extend(content_tokens)

        return ids

    def decode_to_conversation(self, token_ids):
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

    def _parse_chat_text(self, chat_text):
        """
        Parse User/Assistant formatted text back into Conversation object.
        Uses regex to extract role and content from each message block.
        """
        esc_end = re.escape(self.turn_separator)

        # pattern to match "role: content!END"
        pattern = rf"(user|assistant):\s*(.*?){esc_end}"
        matches = re.findall(pattern, chat_text, flags=re.DOTALL)

        if len(matches) == 0:
            raise ValueError(
                f"Parsing failed, did not find any matches. Original text: {chat_text}"
            )

        conversation = Conversation()
        for role, content in matches:
            if role.lower() == "user":
                conversation.add_user_message(content.strip())
            elif role.lower() == "assistant":
                conversation.add_assistant_message(content.strip())
            else:
                raise ValueError(f"Unknown role '{role}' when parsing text")
        return conversation


class DPOConversationPreprocessor:
    """
    TODO add conversation pair validation, similar to what is done in hhrlhf.py
    """
    def __init__(self, tokenizer, max_length=None, turn_separator=_TURN_SEPARATOR):
        self.tokenizer = tokenizer
        self.max_length = (
            tokenizer.model_max_length if max_length is None else max_length
        )
        self.turn_separator = turn_separator
        self.end_tokens = self.tokenizer.encode(turn_separator)


    def __call__(self, conversation_pair):
        conversation_pair = {
            k: self._convert_conversation(v) for k, v in conversation_pair.items()
        }
        return self.encode_to_tokens(conversation_pair)
    
    def encode_to_tokens(self, conversation_pair):
        accepted_dict = self._encode_conversation(conversation_pair["accepted"])
        rejected_dict = self._encode_conversation(conversation_pair["rejected"])
        return {
            "accepted_tokens": accepted_dict,
            "rejected_tokens": rejected_dict,
        }

    def _encode_conversation(self, conversation):
        input_ids = []
        completion_mask = []
        n_messages = len(conversation.messages)
        for i, message in enumerate(conversation.messages, 1):
            mask_idx = 1 if i == n_messages else 0
            msg_ids, msg_mask = self._encode_message(message, mask_idx)
            input_ids.extend(msg_ids)
            completion_mask.extend(msg_mask)

        labels = input_ids[1:][:self.max_length]
        input_ids = input_ids[:-1][:self.max_length]
        completion_mask = completion_mask[1:][:self.max_length]  # matches labels

        return {
            "input_ids": input_ids,
            "labels": labels,
            "completion_mask": completion_mask,
        }

    def _encode_message(self, message, mask_idx):
        input_ids = []
        completion_mask = []

        # role + space
        role_ = message.role + ": "
        role_tokens = self.tokenizer.encode(role_, add_special_tokens=False)
        input_ids.extend(role_tokens)
        completion_mask.extend([0] * len(role_tokens))

        # content
        content_tokens = self.tokenizer.encode(
            message.content, add_special_tokens=False,
        )
        input_ids.extend(content_tokens)
        completion_mask.extend([mask_idx] * len(content_tokens))

        # end
        end_tokens = self.tokenizer.encode(self.turn_separator, add_special_tokens=False)
        input_ids.extend(end_tokens)
        completion_mask.extend([mask_idx] * len(end_tokens))

        # \n
        input_ids.append(self.tokenizer.encode("\n", add_special_tokens=False)[0])
        completion_mask.append(0)

        return input_ids, completion_mask

    def _convert_conversation(self, conversation):
        if not isinstance(conversation, Conversation):
            # handle both dict and LazyRow objects
            if hasattr(conversation, "keys") or hasattr(conversation, "__getitem__"):
                conv_dict = dict(conversation)
                conversation = Conversation.from_dict(conv_dict)
            else:
                raise ValueError(f"Cannot convert {type(conversation)} to Conversation")
        return conversation
