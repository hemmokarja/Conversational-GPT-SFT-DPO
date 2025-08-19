import collections
import contextlib
import copy
import datetime
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import dpo_util
from src.model import FineTuneableGPT2, GPTConfig, LoRAConfig
from src.validation import SFTValidator, DPOValidator

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainerConfig:
    batch_size: int  # split into micro steps
    gradient_acc_steps: int = 10
    log_interval: int = 100
    compile: bool = True
    base_learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    lr_step_size: int = 50_000_000
    lr_gamma: float = 0.33
    weight_decay: float = 1e-5
    betas: Tuple[float] = (0.9, 0.95)
    grad_clip: float = 1.2
    num_workers: Optional[int] = 0
    prefetch_factor: int = None
    pin_memory: bool = False
    validation_samples: int = 1000
    validation_interval: int = 1000
    generate_sample_prompts: List[str] = field(default_factory=list)
    generate_max_tokens: int = 100
    generate_temperature: float = 1.0
    generate_top_k: int = 50
    checkpoint_filepath: Optional[str] = None  # don't save if None

    def __post_init__(self):
        if self.batch_size % self.gradient_acc_steps != 0:
            raise ValueError("batch_size must be divisible by gradient_acc_steps")


@dataclass
class DPOTrainerConfig(SFTTrainerConfig):
    sft_checkpoint_filepath: str = None
    beta: float = 0.1

    def __post_init__(self):
        if self.sft_checkpoint_filepath is None:
            raise ValueError("Must specify sft_checkpoint_filepath")


class _SFTCollator:
    def __init__(self, padding_idx, ignored_idx=-100):
        if padding_idx is None or ignored_idx is None:
            raise ValueError("padding_idx or ignored_idx must not be None")

        self.padding_idx = padding_idx
        self.ignored_idx = ignored_idx

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        labels = []
        valid_masks = []

        for item in batch:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len

            padded_input = item["input_ids"] + [self.padding_idx] * pad_len
            padded_labels = item["labels"] + [self.ignored_idx] * pad_len
            valid_mask = [True] * seq_len + [False] * pad_len

            input_ids.append(padded_input)
            labels.append(padded_labels)
            valid_masks.append(valid_mask)

        return {
            "x": torch.tensor(input_ids),
            "y": torch.tensor(labels),
            "valid_mask": torch.tensor(valid_masks)
        }


class _DPOCollator:
    def __init__(self, padding_idx):
        if padding_idx is None:
            raise ValueError("padding_idx must not be None")
        self.padding_idx = padding_idx

    def __call__(self, batch):
        max_len = max(
            max(len(item["accepted_tokens"]["input_ids"]) for item in batch),
            max(len(item["rejected_tokens"]["input_ids"]) for item in batch)
        )

        result = {"accepted": {}, "rejected": {}}

        for split in ["accepted", "rejected"]:
            split_data = {"x": [], "y": [], "completion_mask": [], "valid_mask": []}
            for item in batch:
                tokens = item[f"{split}_tokens"]

                input_ids = tokens["input_ids"]
                labels = tokens["labels"]
                completion_mask = tokens["completion_mask"]

                pad_len = max_len - len(input_ids)

                split_data["x"].append(input_ids + [self.padding_idx] * pad_len)
                split_data["y"].append(labels + [self.padding_idx] * pad_len)
                split_data["completion_mask"].append(completion_mask + [0] * pad_len)
                split_data["valid_mask"].append([1] * len(input_ids) + [0] * pad_len)

            result[split] = {
                k: torch.tensor(v, dtype=torch.long) for k, v in split_data.items()
            }

        return result


def _get_learning_rate_stepwise(
    samples_seen, base_lr=3e-4, min_lr=1e-6, step_size=50_000_000, gamma=0.33
):
    # reduce lr by a factor of gamma every step_size samples
    # set base_lr==min_lr to effectively turn scheduling off
    if not base_lr >= min_lr:
        raise ValueError(f"Set base_lr {base_lr} equal or greater than min_lr {min_lr}")
    num_steps = samples_seen // step_size
    lr = base_lr * (gamma**num_steps)
    return max(lr, min_lr)


def _configure_optimizer(model, weight_decay, learning_rate, betas):
    if callable(getattr(model, "get_optimizer_parameters", None)):
        params = model.get_optimizer_parameters()
    else:
        logger.warning(
            "Model does not have method get_optimizer_parameters, defaulting to all"
            "parameters that require grad"
        )
        params = [p for p in model.parameters() if p.requires_grad]

    # only >=2D parameters will be weight decayed, i.e. all weight tensors in
    # matmuls + embeddings decay, but biases and layernorms won"t.
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer, params


def _to_hms(took):
    took = int(took)
    hours = took // 3600
    minutes = (took % 3600) // 60
    seconds = took % 60
    return hours, minutes, seconds


def _print_train_results(
    iter_, samples_seen, avg_loss, lr, took_hms, samples_per_sec
):
    h, m, s = took_hms
    print(
        f"🔄 iter: {iter_:>6,} │ "
        f"📊 samples: {samples_seen:>8,} │ "
        f"📉 loss: {avg_loss:>7.4f} │ "
        f"📈 lr: {lr:>9.2e} │ "
        f"⏳ time: {h:02}:{m:02}:{s:02} | "
        f"⚡ {int(samples_per_sec):>4,} samples/s"
    )


def _print_validation_results(metrics, samples, samples_seen, took_hms, mode):
    assert mode in ["sft", "dpo"]

    h, m, s = took_hms
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    print(f"📊 METRICS (samples seen: {samples_seen:,}, {h:02}:{m:02}:{s:02})")
    print("-" * 40)
    print(f"  Loss:        {metrics['loss']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.1%}")

    if mode == "sft":
        print(f"  Perplexity:  {metrics['perplexity']:.2f}")
    elif mode == "dpo":
        print(f"  LogP Margin: {metrics['logprob_margin']:.2f}")

    print()
    print("🤖 SAMPLE COMPLETIONS")
    print("-" * 40)
    for i, sample in enumerate(samples, 1):
        print(f"\n[Sample {i}]")
        print(f"Prompt: {sample['prompt']}")
        print(f"Response: {sample['completion']}")
        if i < len(samples):
            print("-" * 30)

    print("\n" + "="*80 + "\n")


class BaseTrainer(ABC):
    def __init__(
        self, config, model, tokenizer, train_dataset, validation_dataset, device
    ):
        self.config = config
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.micro_batch_size = config.batch_size // config.gradient_acc_steps
        self.samples_seen = 0
        self.best_loss = float("inf")

        optimizer, trainable_params = _configure_optimizer(
            model, config.weight_decay, config.base_learning_rate, config.betas
        )
        self.optimizer = optimizer
        self.trainable_params = trainable_params

        collator = self._get_collator(model.config)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
            shuffle=True,
            collate_fn=collator
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.micro_batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
            shuffle=False,
            collate_fn=collator
        )
        self.train_iterator = iter(self.train_loader)

        if config.compile:
            self.model = torch.compile(self.model)

        self.ctx = (
            contextlib.nullcontext()
            if device.type == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        )

    def _get_next_batch(self, mode="train"):
        if mode == "train":
            iterator = self.train_iterator
        elif mode == "validation":
            iterator = self.validation_iterator
        else:
            raise ValueError(f"Unknown mode '{mode}', expected 'train' or 'validation'")

        try:
            return next(iterator)
        except StopIteration:
            logger.info(f"Exhausted {mode} iterator epoch, restarting from beginning")

            if mode == "train":
                self.train_iterator = iter(self.train_loader)
                return next(self.train_iterator)
            else:
                self.validation_iterator = iter(self.validation_loader)
                return next(self.validation_iterator)

    def _prepare_batch(self, batch):
        """
        Recursively move tensors to devide. Handles lists, tuples, dicts and
        nested dicts.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {key: self._prepare_batch(value) for key, value in batch.items()}
        elif isinstance(batch, (list, tuple)):
            container_type = type(batch)
            return container_type(self._prepare_batch(item) for item in batch)
        else:
            return batch  # ints, strs, etc.

    def _samples_in_batch(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.size(0)
        elif isinstance(batch, dict):
            return max(self._samples_in_batch(b) for b in batch.values())
        elif isinstance(batch, (list, tuple)):
            return max(self._samples_in_batch(b) for b in batch)
        else:
            raise ValueError(
                f"Expcted batch to be a torch.Tensor, dict, list, or tuple, got {batch}"
            )

    def _set_optimizer_lr(self):
        lr = _get_learning_rate_stepwise(
            self.samples_seen,
            base_lr=self.config.base_learning_rate,
            min_lr=self.config.min_learning_rate,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _take_optimisation_step(self):
        total_loss = 0

        self._set_optimizer_lr()

        for _ in range(self.config.gradient_acc_steps):
            batch = self._get_next_batch("train")
            batch = self._prepare_batch(batch)

            with self.ctx:
                forward_output = self._model_forward(batch)
                loss = forward_output["loss"] / self.config.gradient_acc_steps
                loss.backward()
                total_loss += loss.item()
            
            self.samples_seen += self._samples_in_batch(batch)

        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return total_loss

    def _crossed_interval(self, interval):
        this_iter = self.samples_seen // interval
        prev_iter = (self.samples_seen - self.config.batch_size) // interval
        prev_iter = max(prev_iter, 0)
        return this_iter > prev_iter

    def train(self, n_samples):
        logger.info(f"Staring model training for {n_samples} samples...")

        self.model.train()

        n_iter = n_samples // self.config.batch_size + 1

        recent_losses = collections.deque(
            maxlen=max(self.config.log_interval // self.config.batch_size, 1)
        )
        samples_seen_prev = 0
        t0 = time.time()
        t_start = t0

        for i in range(n_iter):
            loss = self._take_optimisation_step()
            recent_losses.append(loss)

            if self._crossed_interval(self.config.log_interval):
                t1 = time.time()
                took = t1 - t0
                t0 = t1

                samples_per_sec = (self.samples_seen - samples_seen_prev) / took
                samples_seen_prev = self.samples_seen

                took_total = t1 - t_start
                took_hms = _to_hms(took_total)

                _print_train_results(
                    i,
                    self.samples_seen,
                    np.mean(recent_losses),
                    self.get_current_lr(),
                    took_hms,
                    samples_per_sec
                )

            if self._crossed_interval(self.config.validation_interval):
                metrics, samples = self._validate()
                took_total = time.time() - t_start
                took_hms = _to_hms(took_total)
                self._print_validation_results(
                    metrics, samples, self.samples_seen, took_hms
                )
                if self.config.checkpoint_filepath and metrics["loss"] < self.best_loss:
                    self.best_loss = metrics["loss"]
                    self._save_checkpoint(metrics)
                t0 = time.time()

        logger.info("Finished model training.")

    def _validate(self):
        self.model.eval()
        self.validation_iterator = iter(self.validation_loader)
        all_batch_metrics = []
        n_iter = self.config.validation_samples // self.micro_batch_size
        for _ in range(n_iter):
            batch = self._get_next_batch("validation")
            batch = self._prepare_batch(batch)

            with self.ctx, torch.no_grad():
                forward_output = self._model_forward(batch)

            batch_metrics = self.validator.compute_batch_metrics(forward_output)
            all_batch_metrics.append(batch_metrics)

        metrics = self.validator.aggregate_metrics(all_batch_metrics)
        samples = self.validator.generate_samples()
        self.model.train()
        return metrics, samples

    def _save_checkpoint(self, validation_metrics=None):
        cp_dir = os.path.dirname(self.config.checkpoint_filepath)
        if cp_dir and cp_dir != ".":
            os.makedirs(cp_dir, exist_ok=True)

        checkpoint = {
            "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_config": asdict(self.config),
            "model_config": asdict(self.model.config),
            "samples_seen": self.samples_seen,
            "validation_metrics": validation_metrics,
            "tokenizer": self.tokenizer,
        }
        torch.save(checkpoint, self.config.checkpoint_filepath)
        logger.info(f"Checkpoint saved to '{self.config.checkpoint_filepath}'")

    @abstractmethod
    def _get_collator(self, model_config):
        pass

    @abstractmethod
    def _model_forward(self, batch):
        # propagate batch through model(s) and return loss
        pass

    @abstractmethod
    def from_checkpoint(cls):
        # load checkpoint for continuing training
        pass
    
    @abstractmethod
    def _print_validation_results(self):
        pass


class SFTTrainer(BaseTrainer):
    def __init__(
        self, config, model, tokenizer, train_dataset, validation_dataset, device
    ):
        super().__init__(
            config, model, tokenizer, train_dataset, validation_dataset, device
        )
        self.validator = SFTValidator(
            self.model,
            self.tokenizer, 
            self.config,
            self.ctx,
            device,
            prevent_tokens=[tokenizer.pad_token_id],
        )

    def _get_collator(self, model_config):
        return _SFTCollator(model_config.padding_idx, model_config.ignored_idx)

    def _model_forward(self, batch):
        logits, loss = self.model(**batch)
        return {"logits": logits, "loss": loss, "y": batch["y"]}

    @staticmethod
    def _print_validation_results(metrics, samples, samples_seen, took_hms):
        _print_validation_results(metrics, samples, samples_seen, took_hms, mode="sft")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        train_dataset,
        validation_dataset,
        device,
        override_config=None,
    ):
        logger.info(
            f"Initializing SFTTrainer from checkpoint saved on "
            f"'{checkpoint['datetime']}'"
        )
        model_config = GPTConfig(**checkpoint["model_config"])
        model = FineTuneableGPT2(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        if override_config is None:
            trainer_config = SFTTrainerConfig(**checkpoint["trainer_config"])
        else:
            trainer_config = override_config

        trainer = cls(
            trainer_config,
            model,
            checkpoint["tokenizer"],
            train_dataset,
            validation_dataset,
            device,
        )
        trainer.samples_seen = checkpoint["samples_seen"]
        if checkpoint["validation_metrics"] is not None:
            trainer.best_loss = checkpoint["validation_metrics"]["loss"]

        return trainer


class DPOTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        model,
        reference_model,
        tokenizer,
        train_dataset,
        validation_dataset,
        device,
    ):
        super().__init__(
            config, model, tokenizer, train_dataset, validation_dataset, device
        )
        self.reference_model = reference_model.to(device)

        if config.compile:
            self.reference_model = torch.compile(self.reference_model)

        self.validator = DPOValidator(
            self.model,
            self.tokenizer, 
            self.config,
            self.ctx,
            device,
            prevent_tokens=[tokenizer.pad_token_id],
        )

    def _get_collator(self, model_config):
        return _DPOCollator(model_config.padding_idx)

    def _model_forward(self, batch):
        accepted, rejected = batch["accepted"], batch["rejected"]

        y_accepted = accepted["y"]
        y_rejected = rejected["y"]
        cmask_accepted = accepted["completion_mask"]
        cmask_rejected = rejected["completion_mask"]

        exclude_keys = {"labels", "completion_mask"}
        accepted_inputs = {k: v for k, v in accepted.items() if k not in exclude_keys}
        rejected_inputs = {k: v for k, v in rejected.items() if k not in exclude_keys}

        logits_accepted, _ = self.model(**accepted_inputs)
        logits_rejected, _ = self.model(**rejected_inputs)

        with torch.no_grad():
            logits_accepted_ref, _ = self.reference_model(**accepted_inputs)
            logits_rejected_ref, _ = self.reference_model(**rejected_inputs)

        loss, logprobs_accepted, logprobs_rejected = dpo_util.dpo_loss(
            logits_accepted,
            logits_rejected,
            logits_accepted_ref,
            logits_rejected_ref,
            y_accepted,
            y_rejected,
            cmask_accepted,
            cmask_rejected,
            self.config.beta,
            return_logprobs=True
        )
        return {
            "loss": loss,
            "logprobs_accepted": logprobs_accepted,
            "logprobs_rejected": logprobs_rejected,
        }

    @staticmethod
    def _print_validation_results(metrics, samples, samples_seen, took_hms):
        _print_validation_results(metrics, samples, samples_seen, took_hms, mode="dpo")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        train_dataset,
        validation_dataset,
        device,
        override_config=None,
    ):
        """Continue a DPO run from checkpoint"""
        logger.info(
            f"Initializing DPOTrainer from checkpoint saved on "
            f"'{checkpoint['datetime']}'"
        )

        sft_checkpoint = torch.load(
            checkpoint["sft_checkpoint_fileapth"],
            weights_only=False,
            map_location="cpu",
        )
        reference_model_config = GPTConfig(**sft_checkpoint["model_config"])
        reference_model = FineTuneableGPT2(reference_model_config)
        reference_model.load_state_dict(sft_checkpoint["model_state_dict"])

        model_config = GPTConfig(**checkpoint["model_config"])
        lora_config = LoRAConfig(**checkpoint["lora_config"])
        model = FineTuneableGPT2(model_config)
        model.apply_lora(lora_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        if override_config is None:
            trainer_config = DPOTrainerConfig(**checkpoint["trainer_config"])
        else:
            trainer_config = override_config

        trainer = cls(
            trainer_config,
            model,
            reference_model,
            checkpoint["tokenizer"],
            train_dataset,
            validation_dataset,
            device,
        )
        trainer.samples_seen = checkpoint["samples_seen"]
        if checkpoint["validation_metrics"] is not None:
            trainer.best_loss = checkpoint["validation_metrics"]["loss"]

        return trainer

    @classmethod
    def init_from_sft_checkpoint(
        cls,
        checkpoint,
        config,
        lora_config,
        train_dataset,
        validation_dataset,
        device,
    ):
        """Iinitialize a new DPO run from SFT checkpoint"""
        logger.info(
            f"Initializing DPOTrainer from checkpoint saved on "
            f"'{checkpoint['datetime']}'"
        )
        model_config = GPTConfig(**checkpoint["model_config"])
        reference_model = FineTuneableGPT2(model_config)
        reference_model.load_state_dict(checkpoint["model_state_dict"])

        model = copy.deepcopy(reference_model)
        model.apply_lora(lora_config)

        return cls(
            config,
            model,
            reference_model,
            checkpoint["tokenizer"],
            train_dataset,
            validation_dataset,
            device,
        )
