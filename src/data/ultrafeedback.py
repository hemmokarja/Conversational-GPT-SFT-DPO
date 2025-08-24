import datasets


def _parse_conversation_pairs(dataset):
    conversation_pairs = []
    for sample in dataset:
        completions = sorted(
            sample["completions"], key=lambda x: x["overall_score"]
        )
        if len(completions) <= 1:
            continue
        accepted = [
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": completions[-1]["response"]}
        ]
        rejected = [
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": completions[0]["response"]}
        ]
        conversation_pair = {
            "accepted": {"messages": accepted},
            "rejected": {"messages": rejected}
        }
        conversation_pairs.append(conversation_pair)
    return conversation_pairs


def load_conversation_pairs(path, name, validation_size=0.05):
    full_dataset = datasets.load_dataset(path, name)
    train_validation_split = (
        full_dataset["train"].train_test_split(test_size=validation_size, seed=42)
    )
    conversations = []
    for split in ["train", "test"]:
        split_pairs = _parse_conversation_pairs(train_validation_split[split])
        conversations.append(split_pairs)
    return conversations
