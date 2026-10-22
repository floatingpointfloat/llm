"""
Train the GPT model.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import GPT, GPTConfig
from prepare_data_txt import prepare_data, block_size

# training settings
batch_size = 64
epochs = 10
learning_rate = 0.0008

# set to True to continue training from the latest checkpoint
resume_training = True

# master switch
ALLOW_TRAINING = True

if not ALLOW_TRAINING:
    raise RuntimeError("Training isn't allowed currently!")

# checkpoint settings
checkpoint_dir = Path("checkpoints")
checkpoint_file = checkpoint_dir / "latest.pt"
checkpoint_interval = 1000


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
start_batch = 0

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

    if "epoch_complete" in checkpoint:
        if checkpoint["epoch_complete"]:
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = checkpoint["epoch"] - 1
            start_batch = checkpoint.get("batch", 0)
    else:
        start_epoch = checkpoint["epoch"]

    print("Resuming from epoch:", start_epoch + 1)
    print("Resuming from batch:", start_batch)


# print model information
print("Vocabulary size:", config.vocab_size)
print("Dataset size:", len(dataset))
print("Batches per epoch:", len(dataloader))
print("Model parameters:", sum(p.numel() for p in model.parameters()))


# create checkpoint directory
checkpoint_dir.mkdir(parents=True, exist_ok=True)


# training loop
for epoch in range(start_epoch, start_epoch + epochs):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if epoch == start_epoch and batch_idx < start_batch:
            continue

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

        if num_batches % 25 == 0:
            print(
                f"----------Epoch: {epoch + 1} | Batch: {batch_idx + 1} | Loss: {total_loss / num_batches}----------"
            )

        if (batch_idx + 1) % checkpoint_interval == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "epoch_complete": False,
                    "config": config.__dict__,
                },
                checkpoint_file,
            )

            print("Checkpoint saved:", checkpoint_file)

    # calculate average loss for this epoch
    if num_batches == 0:
        print(f"Epoch {epoch + 1} already completed.")
        continue

    average_loss = total_loss / num_batches

    print(f"EPOCH {epoch + 1} | " f"LOSS: {average_loss:.4f}")

    # save model and optimizer state
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "batch": 0,
            "epoch_complete": True,
            "config": config.__dict__,
        },
        checkpoint_file,
    )

    print("Checkpoint saved:", checkpoint_file)

    start_batch = 0

print("Training finished!")
