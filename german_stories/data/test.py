import json
from pathlib import Path
from datasets import load_dataset

# Ausgabedatei definieren
output_file = Path("data") / "kreative_geschichten.jsonl"
output_file.parent.mkdir(parents=True, exist_ok=True)

print("Lade deutschen Geschichten- und Autoren-Datensatz herunter...")

# Wir laden den Story-Benchmark von Hugging Face
dataset = load_dataset(
    "lars1234/story_writing_benchmark", split="train", streaming=True
)

print("Konvertiere in Prompt/Completion-Format...")
count = 0

with output_file.open("w", encoding="utf-8") as f:
    for item in dataset:
        # Wir filtern gezielt nach deutschen Einträgen ('de')
        # Der Datensatz nutzt 'prompt' für die Anweisung und 'story' für den fertigen Text
        if item.get("language") == "de" and "prompt" in item and "story" in item:
            json_line = {
                "prompt": item["prompt"].strip(),
                "completion": item["story"].strip(),
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            count += 1

            if count % 100 == 0:
                print(f"{count} kreative Geschichten verarbeitet...")

print(
    f"\nErfolg! {count} rein deutsche Geschichten wurden in '{output_file}' gespeichert!"
)
