import logging
import random

import datasets
from datasets import Dataset

from src.preprocess import DPOPreprocessor

logger = logging.getLogger(__name__)


def _load_samples(validation_size=0.05):
    full_dataset = datasets.load_dataset("openbmb/UltraFeedback")
    train_validation_split = (
        full_dataset["train"].train_test_split(test_size=validation_size, seed=42)
    )

    train_validation_samples = []
    for split in ["train", "test"]:
        samples = []
        for sample in train_validation_split[split]:
            completions = sorted(
                sample["completions"], key=lambda x: x["overall_score"]
            )
            if len(completions) <= 1:
                continue
            samples.append(
                {
                    "prompt": sample["instruction"],
                    "accepted": completions[-1]["response"],
                    "rejected": completions[0]["response"],
                }
            )
        train_validation_samples.append(samples)
    return train_validation_samples


def _length_ok(example, max_len):
    return (
        len(example["accepted_tokens"]["input_ids"]) < max_len
        and len(example["rejected_tokens"]["input_ids"]) < max_len
    )


def load_ultrafeedback_dataset(
    tokenizer, validation_size=0.05, max_len=None, num_proc=8
):
    train_samples, validation_samples = _load_samples(validation_size)
    logger.info(
        f"Loaded {len(train_samples)} train and {len(validation_samples)} "
        "validation samples"
    )

    random.Random(42).shuffle(train_samples)
    random.Random(42).shuffle(validation_samples)

    datasets_ = []
    for split_samples in [train_samples, validation_samples]:
        dataset = Dataset.from_list(split_samples)

        # preprocess sequences to one token too long here so that filter function
        # filters out the overlength sequences
        preprocessor = DPOPreprocessor(tokenizer)
        datasets.logging.disable_progress_bar()
        preprocessed_dataset = dataset.map(preprocessor, num_proc=num_proc)

        before = len(preprocessed_dataset)
        max_len_ = max_len or tokenizer.model_max_length
        filter_fn = lambda x: _length_ok(x, max_len_)
        filtered_dataset = preprocessed_dataset.filter(filter_fn, num_proc=num_proc)
        after = len(filtered_dataset)
        logger.info(
            f"Filtered {before - after} overlength examples ({after}/{before} kept)"
        )

        datasets_.append(filtered_dataset)
    return datasets_
