from datasets import load_dataset
from pathlib import Path

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
)

output_file = Path("data") / "training_text.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)

with output_file.open("w", encoding="utf-8") as file:
    for i, article in enumerate(dataset):
        if i >= 100_000:
            break

        file.write(article["text"])
        file.write("\n\n")

        if i % 1000 == 0:
            print(i)

print("Wikipedia sample saved!")
