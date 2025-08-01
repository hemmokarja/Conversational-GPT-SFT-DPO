import torch


class Collator:
    def __init__(self, tokenizer, ignored_idx=-100):
        self.padding_idx = tokenizer.pad_token_id
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
