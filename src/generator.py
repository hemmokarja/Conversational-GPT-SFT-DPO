import logging

import torch

from src.conversation import Conversation
from src.preprocess import GenerationConversationPreprocessor

logger = logging.getLogger(__name__)


class AssistantResponseGenerator:
    """Class for generating Assistant responses"""

    def __init__(self, model, tokenizer, ctx, device):
        self.model = model
        self.ctx = ctx
        self.device = device
        self.preprocessor = GenerationConversationPreprocessor(
            tokenizer, model.config.ignored_idx
        )

    def _attempt_generation(
        self,
        conversation,
        max_tokens,
        temperature,
        top_k,
        prevent_tokens
    ):
        processed = self.preprocessor(conversation)
        x = torch.tensor(processed["input_ids"], device=self.device).unsqueeze(0)

        with self.ctx, torch.no_grad():
            generated = self.model.generate(
                x,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                end_tokens=self.preprocessor.end_tokens,
                prevent_tokens=prevent_tokens,
            )

        new_conversation = self.preprocessor.decode_to_conversation(generated)
        last_message = new_conversation.messages[-1]  # Fixed: use new_conversation

        if last_message.role != "assistant" or last_message.content is None:
            raise ValueError("Generated message is not a valid assistant response")
        
        if isinstance(last_message.content, str) and last_message.content.strip() == "":
            raise ValueError("Generated assistant message is empty")
        
        return new_conversation
    
    def generate(
        self,
        conversation,
        max_tokens,
        temperature,
        top_k,
        prevent_tokens,
        max_retries=5,
    ):
        if not isinstance(conversation, Conversation):
            raise ValueError(f"Expected Conversation object, got {type(conversation)}")
        
        for attempt in range(max_retries):
            try:
                new_conversation = self._attempt_generation(
                    conversation,
                    max_tokens,
                    temperature,
                    top_k,
                    prevent_tokens
                )
                return new_conversation
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(
                        f"Failed to generate valid response after {max_retries} "
                        f"attempts for conversation {conversation}. Final error: {e}"
                    )
                    return None
                else:
                    logger.debug(
                        f"Generation attempt {attempt + 1} failed, retrying. Error: {e}"
                    )
