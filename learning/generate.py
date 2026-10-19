"""
Generate text with the GPT model
"""

from config import training_file, tokenizer_file, checkpoint_file
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

from pathlib import Path
import torch
import time

# settings for the generation
max_new_tokens = 2000
temperature = 0.7

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


def generate_text(prompt: str, max_new_tokens: int = 100, generated_tokens=0):
    start_time = time.time()  # get the time for the token per second counter

    # version live printing
    print(prompt, end="")

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
            generated_tokens += 1

            # stop after the end of text token
            if next_token.item() == end_of_text_id:
                break

            # version live printing
            print(tokenizer.decode(next_token[0].tolist()), end="")

            # append token
            idx = torch.cat((idx, next_token), dim=1)

        # tokens per second
        computation_time = time.time() - start_time
        tokens_per_second = round(generated_tokens / computation_time, 1)
        print(
            f"\nTokens per second: {tokens_per_second}\nGenerated tokens: {generated_tokens}\nComputation time: {round(computation_time, 1)}s\n"
        )

        # decode generated sequence
        return tokenizer.decode(idx[0].tolist()), tokens_per_second, generated_tokens


def main():
    while True:
        prompt = input("Enter a prompt: ")

        # version printing afterwards
        # generated_text, tokens_per_second, generated_tokens = generate_text(prompt, max_new_tokens)
        # print("\nGenerated text:\n")
        # print(generated_text)

        # version live printing
        print("\nGenerated text:\n")
        generated_text, tokens_per_second, generated_tokens = generate_text(
            prompt, max_new_tokens=max_new_tokens
        )

        again = input("\nGenerate new text? (Y/n): ")
        if again == "n":
            break


if __name__ == "__main__":
    main()
