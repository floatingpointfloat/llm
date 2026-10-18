"""
Prepare training data for the GPT model.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import BPETokenizer

# paths
training_file = Path("data\physics_training_corpus.txt")
tokenizer_file = Path("tokenizer.json")


# training settings
block_size = 128
batch_size = 32


# dataset for next-token prediction
class GPTDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        if len(data) <= block_size:
            raise ValueError("Training data must contain more tokens than block_size.")

        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]

        return x, y


def prepare_data():  # pretty much the only important part - main is used only for debug
    # load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_file)

    # load training text
    text = training_file.read_text(encoding="utf-8")

    # encode text into token ids
    token_ids = tokenizer.encode(text)

    # convert token ids into a pytorch tensor
    data = torch.tensor(token_ids, dtype=torch.long)

    # create dataset
    dataset = GPTDataset(data, block_size)

    return tokenizer, dataset


def main():
    # prepare data
    tokenizer, dataset = prepare_data()

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # debug
    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Dataset size:", len(dataset))

    x, y = dataset[0]

    print("\nSingle example:")
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)

    print("\nFirst 10 input tokens:")
    print(x[:10])

    print("\nFirst 10 target tokens:")
    print(y[:10])

    x, y = next(iter(dataloader))

    print("\nBatch:")
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
