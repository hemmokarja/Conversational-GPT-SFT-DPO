import contextlib
import logging
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass()
class Gpt4RecTrainerConfig:
    compile: bool = True
    gradient_acc_steps: int = 10
    base_learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    lr_step_size: int = 50_000_000
    lr_gamma: float = 0.33
    weight_decay: float = 1e-5
    betas: Tuple[float] = (0.9, 0.95)
    grad_clip: float = 1.2


class _Collator:
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
            valid_mask = [1] * seq_len + [0] * pad_len

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


def _make_iterator(dataset, padding_idx, ignored_idx, batch_size, shuffle=False):
    collator = _Collator(padding_idx, ignored_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )
    return iter(loader)


class Trainer:
    def __init__(
        self, config, model, tokenizer, train_dataset, validation_dataset, device
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.device = device

        optimizer, trainable_params = _configure_optimizer(
            model, config.weight_decay, config.base_learning_rate, config.betas
        )
        self.optimizer = optimizer
        self.trainable_params = trainable_params

        if self.config.compile:
            self.model = torch.compile(self.model)

        self.train_iterator = _make_iterator(
            train_dataset,
            model.config.padding_idx,
            model.config.ignored_idx,
            config.batch_size,
            config.shuffle
        )
        self.train_iterator = _make_iterator(
            validation_dataset,
            model.config.padding_idx,
            model.config.ignored_idx,
            config.batch_size,
            config.shuffle
        )

        self.samples_seen = 0

        self.ctx = (
            contextlib.nullcontext()
            if self.device.type == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        )

    def get_next_batch(self, mode="train"):
        if mode == "train":
            iterator = self.train_iterator
            loader = self.train_loader
        elif mode == "validation":
            iterator = self.validation_iterator
            loader = self.validation_loader
        else:
            raise ValueError(f"Unknown mode: {mode}")

        try:
            return next(iterator)
        except StopIteration:
            logger.info(f"Exhausted {mode} iterator epoch, restarting from beginning")
            new_iterator = iter(loader)

            if mode == "train":
                self.train_iterator = new_iterator
            else:
                self.validation_iterator = new_iterator

            return next(new_iterator)

    def prepare_batch(self, batch):
        return {k: t.to(self.device, non_blocking=True) for k, t in batch.items()}

    def set_optimizer_lr(self):
        lr = _get_learning_rate_stepwise(
            self.samples_seen,
            base_lr=self.config.base_learning_rate,
            min_lr=self.config.min_learning_rate,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # def get_current_lr(self):
    #     return self.optimizer.param_groups[0]["lr"]

    def take_optimisation_step(self):
        total_loss = 0

        self.set_optimizer_lr()

        for _ in range(self.config.gradient_acc_steps):
            batch = self.get_next_batch("train")
            batch = self.prepare_batch(batch)
            
            self.samples_seen += batch["y"].shape[0]

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
