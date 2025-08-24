import logging

import datasets
import numpy as np
from datasets import Dataset
from src.preprocess import SFTConversationPreprocessor, DPOConversationPreprocessor

from src.data import hh_rlhf, oasst, smoltalk, ultrafeedback

logger = logging.getLogger(__name__)
rng = np.random.default_rng(42)
datasets.logging.disable_progress_bar()

SFT_DATASET_DISPATCH_TABLE = {
    "HuggingFaceTB/smoltalk": smoltalk.load_conversations,
    "OpenAssistant/oasst1": oasst.load_conversations,
    "OpenAssistant/oasst2": oasst.load_conversations,
}
DPO_DATASET_DISPATCH_TABLE = {
    "Anthropic/hh-rlhf": hh_rlhf.load_conversation_pairs,
    "openbmb/UltraFeedback": ultrafeedback.load_conversation_pairs,
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


def _load_conversations(dataset_path_names, sampling_factors, dataset_dispatch_table):

    if len(dataset_path_names) != len(sampling_factors):
        raise ValueError(
            "dataset_path_names and sampling_factors must be equal length"
        )

    train_conversations, validation_conversations = [], []
    for idx, path_name in enumerate(dataset_path_names):
        path, name = _normalize_path_name(path_name)
        if path not in dataset_dispatch_table:
            raise ValueError(
                f"Unknown path {path}, supported paths include "
                f"{', '.join(dataset_dispatch_table.keys())}"
            )

        train_convs, validation_convs = dataset_dispatch_table[path](
            path, name
        )
        if sampling_factors is not None:
            factor = sampling_factors[idx]
            if factor != 1.0:
                train_convs = _resample(train_convs, factor)
                validation_convs = _resample(validation_convs, factor)

        path_name_fmt = path
        if name is not None:
            path_name_fmt += (", " + name)
        logger.info(
            f"Loaded {len(train_convs)} train and {len(validation_convs)} "
            f"conversations from '{path_name_fmt}'"
        )

        train_conversations.extend(train_convs)
        validation_conversations.extend(validation_convs)

    rng.shuffle(train_conversations)
    rng.shuffle(validation_conversations)

    return train_conversations, validation_conversations


def _length_ok(example, max_len):
    return (
        len(example["accepted_tokens"]["input_ids"]) < max_len
        and len(example["rejected_tokens"]["input_ids"]) < max_len
    )


def make_sft_datasets(
    dataset_path_names,
    sampling_factors=None,
    tokenizer=None,
    ignored_idx=-100,
    num_proc=8,
):
    train_conversations, validation_conversations = _load_conversations(
        dataset_path_names, sampling_factors, SFT_DATASET_DISPATCH_TABLE
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
        preprocessed_dataset = dataset.map(preprocessor, num_proc=num_proc)
        datasets_.append(preprocessed_dataset)
    return datasets_


def make_dpo_datasets(
    dataset_path_names,
    sampling_factors=None,
    tokenizer=None,
    max_len=None,
    num_proc=8,
):
    train_conversations, validation_conversations = _load_conversations(
        dataset_path_names, sampling_factors, DPO_DATASET_DISPATCH_TABLE
    )
    logger.info(
        f"Loaded in total {len(train_conversations)} train and "
        f"{len(validation_conversations)} validation conversations"
    )
    datasets_ = []
    logger.info("Preprocessing conversations, this might take a while...")
    for conversations in [train_conversations, validation_conversations]:
        dataset = Dataset.from_list(conversations)
        
        preprocessor = DPOConversationPreprocessor(tokenizer)
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
