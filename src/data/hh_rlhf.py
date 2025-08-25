import logging
import re

import datasets

logger = logging.getLogger(__name__)


def _parse_conversation(text):
    """
    Convert a single HH-RLHF formatted string into ChatML format.
    Example:
        Input: "\n\nHuman: Hello\n\nAssistant: Hi there!"
        Output: {
          "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
          ]
        }
    """
    text = text.strip()

    pattern = r"(Human|Assistant):"
    parts = re.split(pattern, text)

    messages = []
    for i in range(1, len(parts), 2):
        role = parts[i].strip()
        content = parts[i+1].strip()

        if role.lower() == "human":
            role_mapped = "user"
        else:
            role_mapped = "assistant"

        messages.append({"role": role_mapped, "content": content})

    return {"messages": messages}


def _parse_conversation_pairs(dataset):
    return [
        {
            "accepted": _parse_conversation(sample["chosen"]),
            "rejected": _parse_conversation(sample["rejected"])
        } for sample in dataset
    ]


def _roles_alternate(messages):
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i-1]["role"]:
            return False
    return True


def _is_valid_conversation_pair(conversation_pair):
    accepted = conversation_pair["accepted"]["messages"]
    rejected = conversation_pair["rejected"]["messages"]

    if accepted[0]["role"] != "user" or rejected[0]["role"] != "user":
        logger.debug("Conversation must start with a user message")
        return False

    if accepted[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        logger.debug("Conversation must end with an assistant message")
        return False

    if len(accepted) != len(rejected):
        logger.debug(
            "Accepted and rejected conversations must be equal in length, got "
            f"{len(accepted)} and {len(rejected)}"
        )
        return False
    
    if not (_roles_alternate(accepted) and _roles_alternate(rejected)):
        logger.debug(
            "Conversation roles must alternate between user and assistant"
        )
        return False

    for acc_msg, rej_msg in zip(accepted[:-1], rejected[:-1]):

        if acc_msg["content"].strip() == "" or rej_msg["content"].strip() == "":
            logger.debug("Message cannot be empty")
            return False

        if acc_msg["content"] != rej_msg["content"]:
            logger.debug(
                "Accepted and rejected conversations can't differ elsewhere but in "
                f"the last message"
            )
            return False

    if accepted[-1]["content"] == rejected[-1]["content"]:
        logger.debug(
            "Accepted and rejected conversations must differ in the last message"
        )
        return False

    return True


def _filter_valid_pairs(conversation_pairs):
    valid_pairs = []
    for conv_pair in conversation_pairs:
        if _is_valid_conversation_pair(conv_pair):
            valid_pairs.append(conv_pair)
    return valid_pairs


def load_conversation_pairs(path, name=None):
    full_dataset = datasets.load_dataset(path, name)
    conversations = []
    for split in ["train", "test"]:
        split_pairs = _parse_conversation_pairs(full_dataset[split])
        valid_pairs = _filter_valid_pairs(split_pairs)
        conversations.append(valid_pairs)
    return conversations
