from datasets import load_dataset
from pathlib import Path

# load german wikipedia articles
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.de",  # german articles
    split="train",
    streaming=True,
)

output_file = Path("data") / "training_text_wiki_de.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)

with output_file.open("w", encoding="utf-8") as file:
    for i, article in enumerate(dataset):
        if i >= 1_000:
            break

        file.write(article["text"])
        file.write("\n\n")

        if i % 100 == 0:
            print(f"Articles extracted: {i}")

print(f"German wikipedia sample saved under {output_file}")
