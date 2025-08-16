import contextlib
import logging
import textwrap
from dataclasses import dataclass

import torch

from src.conversation import Conversation
from src.model import FineTuneableGPT2, GPTConfig
from src.preprocess import ConversationPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    generate_max_tokens: int = 10_000
    temperature: float = 1.0
    top_k: int = 50


class _MessageFormatter:
    def __init__(self, line_width=88):
        self.line_width = line_width
        self.role_to_color = {
            "user": "\033[94m",
            "assistant": "\033[92m",
        }
        self.reset_color = "\033[0m"

    def _format_message(self, message):
        if message.content is None:
            raise ValueError("Cannot format message when content is None")

        color = self.role_to_color[message.role]
        header = f"{color}{'=' * 10} {message.role} {'=' * 10}{self.reset_color}"

        wrapped_content = textwrap.fill(
            message.content, 
            width=self.line_width,
            initial_indent="  ",
            subsequent_indent="  "
        )
    
        return f"\n{header}\n{wrapped_content}\n"

    def print_message(self, message):
        print(self._format_message(message), end='')


class Chat:
    def __init__(self, config, model, tokenizer, device):
        self.config = config
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.conversation = Conversation()
        self.preprocessor = ConversationPreprocessor(tokenizer, model.config.ignored_idx)
        self.ctx = (
            contextlib.nullcontext()
            if device.type == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        )
        self.message_formatter = _MessageFormatter()

    @classmethod
    def from_training_checkpoint(cls, checkpoint, config=None, device=None):
        logger.info(
            "Initializing Chat from training checkpoint saved on "
            f"'{checkpoint['datetime']}'"
        )
        if config is None:
            config = ChatConfig()
        if device is None:
            device = torch.device("cpu")
        model_config = GPTConfig(**checkpoint["model_config"])
        model = FineTuneableGPT2(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(config, model, checkpoint["tokenizer"], device)

    def _display_last_message(self):
        message = self.conversation.messages[-1]
        self.message_formatter.print_message(message)

    def _generate(self, max_retries=5):
        processed = self.preprocessor(self.conversation, for_generation=True)
        x = torch.tensor(processed["input_ids"], device=self.device).unsqueeze(0)

        for i in range(max_retries):

            with self.ctx, torch.no_grad():
                generated = self.model.generate(
                    x,
                    max_tokens=self.config.generate_max_tokens,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    end_tokens=self.preprocessor.end_tokens,
                    prevent_tokens=[self.tokenizer.pad_token_id],
                )

            new_conversation = self.preprocessor.decode_tokens_to_conversation(
                generated
            )

            assistant_response = new_conversation.messages[-1].content
            if assistant_response is not None:
                break

            if i == max_retries - 1:
                print(
                    f"Assistant failed to generate response after {max_retries} "
                    "retries. You can try sending another message."
                )
                self.conversation.delete_last_message()  # pop last user message
                return new_conversation, False
        return new_conversation, True

    def reset(self):
        self.conversation = Conversation()

    def chat(self, text, max_retries=5):
        self.conversation.add_user_message(text)
        self._display_last_message()
        self.conversation.add_assistant_message(None)
        new_conversation, success = self._generate(max_retries)
        if success:
            self.conversation = new_conversation
            self._display_last_message()
            return True
        return False
