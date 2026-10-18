import json
from pathlib import Path

input_file = Path("train.jsonl")
output_file = Path("qa_training.txt")

max_examples = 5000

with input_file.open("r", encoding="utf-8") as file:
    with output_file.open("w", encoding="utf-8") as output:
        for index, line in enumerate(file):
            if index >= max_examples:
                break

            if not line.strip():
                continue

            example = json.loads(line)

            for turn in example["turns"]:
                role = turn["role"]
                content = turn["content"]

                if role == "user":
                    output.write(f"User: {content}\n")
                elif role == "model":
                    output.write(f"Assistant: {content}\n")

            # separate conversations
            output.write("\n<|endoftext|>\n\n")

print(f"Conversion completed: {output_file}")
