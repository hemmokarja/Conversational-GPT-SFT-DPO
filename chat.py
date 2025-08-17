import logging

import torch

from src.chat import Chat, ChatConfig

logging.basicConfig(level=logging.INFO)

CHECKPOINT_PATH = "checkpoints/checkpoint-medium-pooled.pt"


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")

    config = ChatConfig(generate_max_tokens=10_000, temperature=1.0, top_k=50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chat = Chat.from_training_checkpoint(checkpoint, config, device)

    print(
        "Chat started! Type `/new` to start a new chat. Press Ctrl+C to exit."
    )
    try:
        while True:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            # clear the input line by moving cursor up and clearing
            print("\033[A\033[K", end="")

            if user_input.lower() == "/new":
                chat.reset()
                print("New chat started!")
                continue

            chat.chat(user_input)

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except EOFError:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
