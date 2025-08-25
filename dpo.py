import logging

import torch

from src.data import dataset
from src.model import LoRAConfig
from src.trainer import DPOTrainer, DPOTrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = None  # if None, start from SFT checkpoint
N_SAMPLES_TRAIN = 124_000

DATASET_PATH_NAMES = [
    "Anthropic/hh-rlhf",
    "openbmb/UltraFeedback",
]
SAMPLING_FACTORS = [0.5, 0.5]
MAX_LEN = 512

DEFAULT_TRAINER_CONFIG = DPOTrainerConfig(
    batch_size=128,
    gradient_acc_steps=32,
    log_interval=8,
    compile=False,
    base_learning_rate=1e-5,
    min_learning_rate=1e-6,
    lr_step_size=2000,
    lr_gamma=0.75,
    weight_decay=0.01,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    num_workers=1,
    prefetch_factor=2,
    pin_memory=False,
    validation_samples=1500,
    validation_interval=5_000,
    generate_sample_prompts=[
        "Can you give me instructions for making lasagna?",
        "I'm travelling to Rome, Italy next summer, what are the best attractions there?",
        "What does a data scientist do?"
    ],
    generate_max_tokens=1024,
    generate_temperature=1.0,
    generate_top_k=50,
    checkpoint_filepath="checkpoints/dpo/medium.pt",
    sft_checkpoint_filepath="checkpoints/sft/medium.pt",
    beta=0.01,
)
# LORA_CONFIG = LoRAConfig(
#     r=32,
#     alpha=32
# )
LORA_CONFIG = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_trainer_from_sft_checkpoint():
    logger.info("Starting a new DPO run from a SFT checkpoint")

    checkpoint = torch.load(
        DEFAULT_TRAINER_CONFIG.sft_checkpoint_filepath,
        weights_only=False,
        map_location="cpu"
    )
    train_dataset, validation_dataset = dataset.make_dpo_datasets(
        DATASET_PATH_NAMES, SAMPLING_FACTORS, checkpoint["tokenizer"], MAX_LEN
    )
    trainer = DPOTrainer.init_from_sft_checkpoint(
        checkpoint,
        DEFAULT_TRAINER_CONFIG,
        LORA_CONFIG,
        train_dataset,
        validation_dataset,
        DEVICE,
    )
    return trainer


def initialize_trainer_from_dpo_checkpoint():
    logger.info("Continuing DPO run from a checkpoint")

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    train_dataset, validation_dataset = dataset.make_dpo_datasets(
        DATASET_PATH_NAMES, SAMPLING_FACTORS, checkpoint["tokenizer"], MAX_LEN
    )
    trainer = DPOTrainer.from_checkpoint(
        checkpoint,
        train_dataset,
        validation_dataset,
        DEVICE,
    )
    return trainer


def main():
    if CHECKPOINT_PATH is None:
        trainer = initialize_trainer_from_sft_checkpoint()
    else:
        trainer = initialize_trainer_from_dpo_checkpoint()
    trainer.train(N_SAMPLES_TRAIN)


if __name__ == "__main__":
    main()
