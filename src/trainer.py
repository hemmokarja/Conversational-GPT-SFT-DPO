import contextlib
import collections
import datetime
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model import FineTuneableGPT2, GPTConfig
from src.validation import SFTValidator

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
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


def _make_loader(
    dataset, padding_idx, ignored_idx, config, micro_batch_size, shuffle=False
):
    collator = _SFTCollator(padding_idx, ignored_idx)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        shuffle=shuffle,
        collate_fn=collator
    )


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


def _print_validation_results(metrics, samples, samples_seen, took_hms):
    h, m, s = took_hms
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    print(f"📊 METRICS (samples seen: {samples_seen:,}, {h:02}:{m:02}:{s:02})")
    print("-" * 40)
    print(f"  Loss:       {metrics['loss']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.1%}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
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


class Trainer:
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

        self.train_loader = _make_loader(
            train_dataset,
            model.config.padding_idx,
            model.config.ignored_idx,
            config,
            self.micro_batch_size,
            shuffle=True
        )
        self.validation_loader = _make_loader(
            validation_dataset,
            model.config.padding_idx,
            model.config.ignored_idx,
            config,
            self.micro_batch_size,
            shuffle=False
        )
        self.train_iterator = iter(self.train_loader)

        if config.compile:
            self.model = torch.compile(self.model)

        self.ctx = (
            contextlib.nullcontext()
            if device.type == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        )

        self.validator = SFTValidator(
            self.model,
            self.tokenizer, 
            self.config,
            self.ctx,
            device,
            prevent_tokens=[tokenizer.pad_token_id],
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
        return {k: t.to(self.device, non_blocking=True) for k, t in batch.items()}

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

            self.samples_seen += batch["y"].size(0)

            with self.ctx:
                _, loss = self.model(**batch)
                loss = loss / self.config.gradient_acc_steps
                loss.backward()
                total_loss += loss.item()

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

    def train(self, n_samples):
        logger.info(f"Staring model training for {n_samples} samples...")

        self.model.train()

        n_iter = n_samples // self.config.batch_size + 1

        recent_losses = collections.deque(maxlen=self.config.log_interval)
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
                metrics, samples = self.validate()
                took_total = time.time() - t_start
                took_hms = _to_hms(took_total)
                _print_validation_results(
                    metrics, samples, self.samples_seen, took_hms
                )
                if self.config.checkpoint_filepath and metrics["loss"] < self.best_loss:
                    self.best_loss = metrics["loss"]
                    self._save_checkpoint(metrics)
                t0 = time.time()

        logger.info("Finished model training.")

    def validate(self):
        self.model.eval()
        self.validation_iterator = iter(self.validation_loader)
        all_batch_metrics = []
        n_iter = self.config.validation_samples // self.micro_batch_size
        for _ in range(n_iter):
            batch = self._get_next_batch("validation")
            batch = self._prepare_batch(batch)
            batch_metrics = self.validator.compute_batch_metrics(batch)
            all_batch_metrics.append(batch_metrics)

        metrics = self.validator.aggregate_metrics(all_batch_metrics)
        samples = self.validator.generate_samples()
        self.model.train()
        return metrics, samples

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        train_dataset,
        validation_dataset,
        device,
        model_cls=FineTuneableGPT2,
    ):
        logger.info(
            f"Initializing Trainer from checkpoint saved on '{checkpoint['datetime']}'"
        )
        model_config = GPTConfig(**checkpoint["model_config"])
        model = model_cls(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        trainer_config = TrainerConfig(**checkpoint["trainer_config"])
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
