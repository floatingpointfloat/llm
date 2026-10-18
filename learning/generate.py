"""
Generate text with the GPT model
"""

from config import training_file, tokenizer_file, checkpoint_file
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

from pathlib import Path
import torch

# settings for the generation
max_new_tokens = 1000
temperature = 0.8

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load the tokenizer
tokenizer = BPETokenizer.load(tokenizer_file)

# load checkpoint
checkpoint = torch.load(
    checkpoint_file,
    map_location=device,
    weights_only=True,
)

# restore the config
config = GPTConfig(**checkpoint["config"])

# load the model
model = GPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])  # load the parameters

# swith to eval mode
model.eval()

# get the end of text token id
end_of_text_id = tokenizer.tokenizer.token_to_id("<|endoftext|>")


def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    input_ids = tokenizer.encode(prompt)

    if not input_ids:  # prevent the user from entering without having entered something
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
            # keep the input within the models context 'window' size
            idx_cond = idx[:, -config.block_size :]

            # get the prediction
            logits = model(idx_cond)

            # get logits for the last position
            logits = logits[:, -1, :]

            # apply the temperature
            logits = logits / temperature

            # convert logits to probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # sample next token
            next_token = torch.multinomial(probabilities, num_samples=1)

            # stop after the end of text token
            if next_token.item() == end_of_text_id:
                break

            # append token
            idx = torch.cat((idx, next_token), dim=1)

        # decode generated sequence
        return tokenizer.decode(idx[0].tolist())


def main():
    while True:
        prompt = input("Enter a prompt: ")
        prompt = f"User: {prompt}\nAssistant: "
        generated_text = generate_text(prompt, max_new_tokens)

        print("\nGenerated text:\n")
        print(generated_text)

        again = input("\nGenerate new text? (Y/n): ")
        if again == "n":
            break


if __name__ == "__main__":
    main()
