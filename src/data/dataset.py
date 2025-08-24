import logging

import datasets
import numpy as np
from datasets import Dataset
from src.preprocess import SFTConversationPreprocessor

from src.data import oasst, smoltalk

logger = logging.getLogger(__name__)
rng = np.random.default_rng(42)

SFT_DATASET_DISPATCH_TABLE = {
    "HuggingFaceTB/smoltalk": smoltalk.load_conversations,
    "OpenAssistant/oasst1": oasst.load_conversations,
    "OpenAssistant/oasst2": oasst.load_conversations,
}


def _resample(convs, factor):
    n = len(convs)
    target_size = int(round(factor * n))
    indices = rng.choice(n, size=target_size, replace=(factor > 1))
    return [convs[i] for i in indices]


def _normalize_path_name(item):
    if isinstance(item, str):
        item = (item,)
    if len(item) == 1:
        return item[0], None
    elif len(item) == 2:
        return item
    else:
        raise ValueError(
            f"Expected dataset path-name to be str, 1-tuple, or 2-tuple, got {item}"
        )


def _load_conversations(dataset_path_names, sampling_factors):

    if len(dataset_path_names) != len(sampling_factors):
        raise ValueError(
            "dataset_path_names and sampling_factors must be equal length"
        )

    train_conversations, validation_conversations = [], []
    for idx, path_name in enumerate(dataset_path_names):
        path, name = _normalize_path_name(path_name)
        if path not in SFT_DATASET_DISPATCH_TABLE:
            raise ValueError(
                f"Unknown path {path}, supported paths include "
                f"{', '.join(SFT_DATASET_DISPATCH_TABLE.keys())}"
            )

        train_convs, validation_convs = SFT_DATASET_DISPATCH_TABLE[path](
            path, name
        )
        if sampling_factors is not None:
            factor = sampling_factors[idx]
            if factor != 1.0:
                train_convs = _resample(train_convs, factor)
                validation_convs = _resample(validation_convs, factor)
        
        logger.info(
            f"Loaded {len(train_convs)} train and {len(validation_convs)} "
            f"conversations from '{path}'"
        )

        train_conversations.extend(train_convs)
        validation_conversations.extend(validation_convs)

    rng.shuffle(train_conversations)
    rng.shuffle(validation_conversations)

    return train_conversations, validation_conversations


def make_sft_datasets(
    dataset_path_names, tokenizer, sampling_factors=None, ignored_idx=-100, num_proc=8
):
    train_conversations, validation_conversations = _load_conversations(
        dataset_path_names, sampling_factors
    )
    logger.info(
        f"Loaded in total {len(train_conversations)} train and "
        f"{len(validation_conversations)} validation conversations"
    )
    datasets_ = []
    logger.info("Preprocessing conversations, this might take a while...")
    for conversations in [train_conversations, validation_conversations]:
        dataset = Dataset.from_list(conversations)
        preprocessor = SFTConversationPreprocessor(tokenizer, ignored_idx)
        datasets.logging.disable_progress_bar()
        preprocessed_dataset = dataset.map(preprocessor, num_proc=num_proc)
        datasets_.append(preprocessed_dataset)
    return datasets_
