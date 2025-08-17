import logging
from abc import ABC, abstractmethod

import torch
import numpy as np

from src.generator import AssistantResponseGenerator
from src.preprocess import ConversationPreprocessor, Conversation

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

        self.preprocessor = ConversationPreprocessor(
            self.tokenizer, self.model.config.ignored_idx
        )
        self.generator = AssistantResponseGenerator(
            self.model, self.preprocessor, self.ctx, self.device
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


class SFTValidator(BaseValidator):
    def __init__(
        self, model, tokenizer, trainer_config, ctx, device, prevent_tokens=None
    ):
        super().__init__(model, tokenizer, trainer_config, ctx, device, prevent_tokens)

    def compute_batch_metrics(self, batch):
        with self.ctx, torch.no_grad():
            logits, loss = self.model(**batch)  # [B, S, vocab], scalar

        preds = logits.argmax(dim=-1)  # [B, S]

        preds_flat = preds.view(-1)
        y_flat = batch["y"].view(-1)

        mask = y_flat != self.model.config.ignored_idx
        correct = (y_flat == preds_flat) & mask
        accuracy = correct.sum().item() / mask.sum().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "perplexity": np.exp(loss.item())
        }

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
                end_tokens=self.preprocessor.end_tokens,
                prevent_tokens=self.prevent_tokens,
                max_retries=max_retries
            )
            if new_conversation is None:
                continue

            completion = new_conversation.messages[-1].content
            samples.append({"prompt": prompt, "completion": completion})

        return samples


# class DPOValidator(BaseValidator):
#     def compute_batch_metrics(self, batch, model_output):
#         policy_chosen_logps, policy_rejected_logps, loss = model_output
        
#         # DPO-specific metrics
#         reward_margin = policy_chosen_logps - policy_rejected_logps
#         accuracy = (reward_margin > 0).float().mean().item()
        
#         return {
#             "loss": loss.item(),
#             "accuracy": accuracy,
#             "reward_margin": reward_margin.mean().item()
#         }

#     def generate_samples(self, model, tokenizer, device, n_samples=5):
#         samples = []
#         sample_prompts = self._get_sample_prompts(n_samples)

#         for prompt in sample_prompts:
#             input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
#             responses = []
#             for _ in range(2):
#                 with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
#                     generated = model.generate(
#                         input_ids,
#                         max_new_tokens=100,
#                         temperature=0.8,
#                         do_sample=True,
#                         pad_token_id=tokenizer.pad_token_id
#                     )

#                 response = tokenizer.decode(
#                     generated[0][len(input_ids[0]):], 
#                     skip_special_tokens=True
#                 )
#                 responses.append(response)
            
#             samples.append({
#                 "prompt": prompt,
#                 "response_a": responses[0],
#                 "response_b": responses[1]
#             })
        
#         return samples
    
#     def _get_sample_prompts(self, n_samples):
#         return ["Sample prompt " + str(i) for i in range(n_samples)]
