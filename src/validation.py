from abc import ABC, abstractmethod

import torch
import numpy as np

from src import text_util
from src.preprocess import ConversationPreprocessor


def _make_conversation_for_sample_generation(user_prompt):
    conversation = text_util.make_user_conversation(user_prompt)
    conversation = text_util.add_assistant_message(conversation, assistant_content=None)
    return conversation


class BaseValidator(ABC):    
    def __init__(
        self,
        model,
        tokenizer,
        trainer_config,
        ctx,
        device,
        prevent_tokens=None,
        stop_tokens=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_config = trainer_config
        self.ctx = ctx
        self.device = device
        self.prevent_tokens = prevent_tokens
        self.stop_tokens = stop_tokens

        self.preprocessor = ConversationPreprocessor(
            tokenizer, model.config.ignored_idx
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
        self,
        model,
        tokenizer,
        trainer_config,
        ctx,
        device,
        prevent_tokens=None,
        stop_tokens=None,
    ):
        super().__init__(
            model, tokenizer, trainer_config, ctx, device, prevent_tokens, stop_tokens
        )

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

    def generate_samples(self):
        samples = []

        for prompt in self.trainer_config.sample_prompts:

            conversation = _make_conversation_for_sample_generation(prompt)
            processed = self.preprocessor(conversation, for_generation=True)
            x = torch.tensor(processed["input_ids"], device=self.device).unsqueeze(0)

            with self.ctx, torch.no_grad():
                generated = self.model.generate(
                    x,
                    max_tokens=100,  # TODO config
                    temperature=1.0,  # TODO config
                    top_k=50,  # TODO config
                    stop_tokens=self.stop_tokens,
                    prevent_tokens=self.prevent_tokens,
                )

            conversation = self.preprocessor.decode_tokens_to_conversation(generated)
            completion = text_util.get_last_assistant_message_content(conversation)
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
