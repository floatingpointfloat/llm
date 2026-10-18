import json
from pathlib import Path

input_file = Path("train.jsonl")
output_file = Path("qa_training.txt")

with input_file.open("r", encoding="utf-8") as file:
    lines = file.readlines()

with output_file.open("w", encoding="utf-8") as file:
    for line in lines:
        if not line.strip():
            continue

        example = json.loads(line)

        # inspect the dataset structure before adapting this part
        print(example)
        break
