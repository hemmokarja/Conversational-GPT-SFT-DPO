import logging

from src.data import oasst
import torch
from transformers import GPT2Tokenizer

from src import text_util
from src.model import GPT2
from src.trainer import SFTTrainer, SFTTrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "checkpoints/checkpoint-medium-pooled.pt"  # if None, start from scratch
N_SAMPLES_TRAIN = 30_100

BASE_MODEL = "gpt2-medium"
OASST_VERSION = "pooled"

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

DEFAULT_TRAINER_CONFIG = SFTTrainerConfig(
    batch_size=64,
    gradient_acc_steps=8,
    log_interval=8,
    compile=False,
    base_learning_rate=1e-4,
    min_learning_rate=1e-6,
    lr_step_size=600,
    lr_gamma=0.75,
    weight_decay=0.01,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    num_workers=1,
    prefetch_factor=2,
    pin_memory=False,
    validation_samples=1000,
    validation_interval=2_000,
    generate_sample_prompts=[
        "Can you give me instructions for making lasagna?",
        "I'm travelling to Rome, Italy next summer, what are the best attractions there?",
        "What does a data scientist do?"
    ],
    generate_max_tokens=10_000,
    generate_temperature=1.0,
    generate_top_k=50,
    checkpoint_filepath="checkpoints/checkpoint-medium-pooled.pt"
)


def initialize_trainer_from_scratch():
    logger.info("Starting a new SFT run from a pre-trained GPT2")

    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
    text_util.add_pad_token_to_tokenizer(tokenizer)

    train_dataset, validation_dataset = oasst.load_oasst_dataset(
        OASST_VERSION, tokenizer
    )

    model = GPT2.from_pretrained(BASE_MODEL, override_args={"dropout": 0.1})
    fine_tuneable = model.to_fine_tuneable()
    fine_tuneable.add_padding_token()

    trainer =  SFTTrainer(
        DEFAULT_TRAINER_CONFIG,
        fine_tuneable,
        tokenizer,
        train_dataset,
        validation_dataset,
        DEVICE,
    )
    return trainer


def initalize_trainer_from_checkpoint():
    logger.info("Continuing SFT run from a checkpoint")

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    train_dataset, validation_dataset = oasst.load_oasst_dataset(
        OASST_VERSION, checkpoint["tokenizer"]
    )
    trainer = SFTTrainer.from_checkpoint(
        checkpoint,
        train_dataset,
        validation_dataset,
        DEVICE,
    )
    return trainer


def main():
    if CHECKPOINT_PATH is None:
        trainer = initialize_trainer_from_scratch()
    else:
        trainer = initalize_trainer_from_checkpoint()
    trainer.train(N_SAMPLES_TRAIN)


if __name__ == "__main__":
    main()
