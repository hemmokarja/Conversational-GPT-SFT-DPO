from collections import defaultdict

import datasets
from datasets import Dataset

from src import data_util


def _parse_valid_conversations_from_tree(
    tree, language="en", require_full_chain=True, require_role_alteration=True
):
    """
    Returns list of conversation paths (root→leaf) where all messages in the path are 
    of desired language.
    - language: desired language of all messages in conversation
    - require_full_chain: if True, discard paths where ancestry is broken
                          (parent missing).
    - require_role_alteration: if True, validate role alternation (e.g., 'user' then
                               'assistant').
    """
    msg_id_to_msg = {msg["message_id"]: msg for msg in tree}

    # walk from leaf to root
    def _get_path(msg_id):
        path = []
        visited = set()
        current_id = msg_id

        while current_id is not None:

            if current_id in visited:
                # cycle detected; abort this path
                return None
            
            visited.add(current_id)

            current = msg_id_to_msg.get(current_id)

            if current is None:
                if require_full_chain:
                    return None  # broken ancestry
                else:
                    break  # stop here, keep what we have

            path.append(current)
            parent_id = current.get("parent_id")
            current_id = parent_id if parent_id else None

        return list(reversed(path))  # root → leaf

    # identify leaves: messages that are not parents of any other
    parent_ids = {msg.get("parent_id") for msg in tree if msg.get("parent_id")}
    all_ids = set(msg_id_to_msg.keys())
    leaf_ids = all_ids - parent_ids

    conversations = []
    for leaf_id in leaf_ids:
        path = _get_path(leaf_id)
        if not path or len(path) < 2:
            continue  # need at least two turns

        # filter to language-only path
        if any(msg.get("lang") != language for msg in path):
            continue

        # validate role alternation if role_field given
        if require_role_alteration:
            roles = [msg.get("role") for msg in path]
            if any(r1 == r2 for r1, r2 in zip(roles, roles[1:])):
                continue

        conversations.append(path)

    return conversations


def _format_conversations(conversation):
    formatted = []
    for msg in conversation:
        role = "user" if msg["role"] == "prompter" else "assistant"
        formatted.append(
            {
                "role": role,
                "content": msg["text"]
            }
        )
    return {"messages": formatted}


def _parse_conversations(dataset):

    # group messages by conversation tree id
    trees = defaultdict(list)
    for msg in dataset:
        trees[msg["message_tree_id"]].append(msg)

    # parse conversation message paths into chronological order
    conversations = []
    for tree in trees.values():
        conversations.extend(_parse_valid_conversations_from_tree(tree))
    
    # format to conventional structure
    conversations = [
        _format_conversations(conversation) for conversation in conversations
    ]

    print(f"Extracted and parsed {len(conversations)} conversations")

    return conversations


def load_oasst_dataset(name, tokenizer, ignored_idx=-100, num_proc=4):
    if name not in ["oasst1", "oasst2"]:
        raise ValueError("name must be 'oasst1' or 'oasst2'")
    
    full_dataset = datasets.load_dataset(f"OpenAssistant/{name}")

    datasets_ = []
    for split in ["train", "validation"]:
        conversations = _parse_conversations(full_dataset[split])
        dataset = Dataset.from_list(conversations)
        datasets.logging.disable_progress_bar()
        preprocessed_dataset = dataset.map(
            data_util.preprocess_fn,
            fn_kwargs={"tokenizer": tokenizer, "ignored_idx": ignored_idx},
            num_proc=num_proc,
        )
        datasets_.append(preprocessed_dataset)
    return datasets_