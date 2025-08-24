import datasets


def _parse_conversations(dataset):
    return [{"messages": msg["messages"]} for msg in dataset]


def load_conversations(path, name=None):
    full_dataset = datasets.load_dataset(path, name)
    conversations = []
    for split in ["train", "test"]:
        split_conversations = _parse_conversations(full_dataset[split])
        conversations.append(split_conversations)
    return conversations
