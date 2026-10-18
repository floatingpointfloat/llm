"""
Generate text with the trained GPT model.
"""

from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer

# settings
max_new_tokens = 1000
temperature = 0.8

# paths
tokenizer_file = Path("tokenizer.json")
checkpoint_file = Path("checkpoints/latest.pt")


# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# load tokenizer
tokenizer = BPETokenizer.load(tokenizer_file)


# load checkpoint
checkpoint = torch.load(
    checkpoint_file,
    map_location=device,
    weights_only=True,
)

# restore model configuration
config = GPTConfig(**checkpoint["config"])

# initialize model and load trained parameters
model = GPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

# switch to evaluation mode
model.eval()


def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    # encode prompt
    input_ids = tokenizer.encode(prompt)

    if not input_ids:
        raise ValueError("The prompt must contain at least one token.")

    # convert to tensor
    idx = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    # generate new tokens
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # keep the input within the model's context window
            idx_cond = idx[:, -config.block_size :]

            # predict next token
            logits = model(idx_cond)

            # get logits for the last position
            logits = logits[:, -1, :]

            # apply temperature
            logits = logits / temperature

            # convert logits to probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # sample next token
            next_token = torch.multinomial(probabilities, num_samples=1)

            # append token
            idx = torch.cat((idx, next_token), dim=1)

    # decode generated sequence
    return tokenizer.decode(idx[0].tolist())


def main():
    prompt = input("Enter a prompt: ")

    generated_text = generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
    )

    print("\nGenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()
