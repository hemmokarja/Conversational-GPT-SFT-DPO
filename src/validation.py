import logging
from abc import ABC, abstractmethod

import torch
import numpy as np

from src.generator import AssistantResponseGenerator
from src.preprocess import Conversation

logger = logging.getLogger(__name__)


class BaseValidator(ABC):    
    def __init__(
        self,
        model,
        tokenizer,
        trainer_config,
        ctx,
        device,
        prevent_tokens=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_config = trainer_config
        self.ctx = ctx
        self.device = device
        self.prevent_tokens = prevent_tokens

        self.generator = AssistantResponseGenerator(
            self.model, self.tokenizer, self.ctx, self.device
        )

    @abstractmethod
    def compute_batch_metrics(self, batch):
        """Process a single batch and return metrics"""
        pass

    @abstractmethod
    def generate_samples(self):
        """Generate sample outputs"""
        pass

    def aggregate_metrics(self, batch_metrics):
        """Aggregate metrics across batches"""
        aggregated = {}
        for key in batch_metrics[0].keys():
            values = [metrics[key] for metrics in batch_metrics]
            aggregated[key] = np.mean(values)
        return aggregated

    def generate_samples(self, max_retries=5):
        samples = []

        for prompt in self.trainer_config.generate_sample_prompts:
            conversation = Conversation()
            conversation.add_user_message(prompt)
            conversation.add_assistant_message(None)

            new_conversation = self.generator.generate(
                conversation,
                max_tokens=self.trainer_config.generate_max_tokens,
                temperature=self.trainer_config.generate_temperature,
                top_k=self.trainer_config.generate_top_k,
                prevent_tokens=self.prevent_tokens,
                max_retries=max_retries
            )
            if new_conversation is None:
                continue

            completion = new_conversation.messages[-1].content
            samples.append({"prompt": prompt, "completion": completion})

        return samples


class SFTValidator(BaseValidator):
    def __init__(
        self, model, tokenizer, trainer_config, ctx, device, prevent_tokens=None
    ):
        super().__init__(model, tokenizer, trainer_config, ctx, device, prevent_tokens)

    def compute_batch_metrics(self, forward_output):
        preds = loss["logits"].argmax(dim=-1)  # [B, S]
        preds_flat = preds.view(-1)

        y_flat = forward_output["y"].view(-1)

        mask = y_flat != self.model.config.ignored_idx
        correct = (y_flat == preds_flat) & mask
        accuracy = correct.sum().item() / mask.sum().item()

        loss = forward_output["loss"].item()
        return {
            "loss": loss,
            "accuracy": accuracy,
            "perplexity": np.exp(loss)
        }


class DPOValidator(BaseValidator):
    def __init__(
        self,
        model,
        tokenizer,
        trainer_config,
        ctx,
        device,
        prevent_tokens=None,
    ):
        super().__init__(model, tokenizer, trainer_config, ctx, device, prevent_tokens)

    def compute_batch_metrics(self, forward_output):
        logprob_margin = (
            forward_output["logprobs_accepted"] - forward_output["logprobs_rejected"]
        )
        accuracy = (logprob_margin > 0).float().mean().item()
        return {
            "loss": forward_output["loss"].item(),
            "accuracy": accuracy,
            "logprob_margin": logprob_margin.mean().item()
        }
