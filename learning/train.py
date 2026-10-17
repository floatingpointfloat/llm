"""
Train the GPT model.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import GPT, GPTConfig
from prepare_data import prepare_data, block_size

# training settings
batch_size = 32
epochs = 10
learning_rate = 3e-4

# set to True to continue training from the latest checkpoint
resume_training = False

# checkpoint settings
checkpoint_dir = Path("checkpoints")
checkpoint_file = checkpoint_dir / "latest.pt"


# select the device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# load training data
tokenizer, dataset = prepare_data()

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


# initialize model
config = GPTConfig(
    vocab_size=tokenizer.get_vocab_size(),
    block_size=block_size,
)

model = GPT(config).to(device)


# initialize optimizer
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
)


# resume training if requested
start_epoch = 0

if resume_training:
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )

    # restore model and optimizer
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # continue with the next epoch
    start_epoch = checkpoint["epoch"]

    print("Resuming from epoch:", start_epoch)


# print model information
print("Vocabulary size:", config.vocab_size)
print("Dataset size:", len(dataset))
print("Model parameters:", sum(p.numel() for p in model.parameters()))


# create checkpoint directory
checkpoint_dir.mkdir(parents=True, exist_ok=True)


# training loop
for epoch in range(start_epoch, start_epoch + epochs):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        # move data to the selected device
        x = x.to(device)
        y = y.to(device)

        # reset gradients from the previous batch
        optimizer.zero_grad()

        # predict the next tokens
        logits = model(x)

        # calculate the loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )

        # calculate gradients
        loss.backward()

        # update model parameters
        optimizer.step()

        # accumulate loss for logging
        total_loss += loss.item()
        num_batches += 1

    # calculate average loss for this epoch
    average_loss = total_loss / num_batches

    print(f"Epoch {epoch + 1} | " f"Loss: {average_loss:.4f}")

    # save model and optimizer state
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": config.__dict__,
        },
        checkpoint_file,
    )

    print("Checkpoint saved:", checkpoint_file)


print("Training finished!")
