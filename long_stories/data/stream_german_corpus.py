import re
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm  # debug information while downloading


#using the dataset german commons
def export_premium_german_corpus(
    output_file="training_text.txt",
    dataset_path="coral-nlp/german-commons",
    max_documents_per_subset=1500,
):
    print("Starting the download of the german corpus")

    end_of_text_token = "<|endoftext|>"

    targets = [
        ("news", "train"),  
        ("cultural", "germanpd"),  
    ]

    successfully_loaded = []
    print("Writing file...")

    with open(output_file, "w", encoding="utf-8") as f:
        for config_name, source_split in targets:
            try:
                count = 0
                print(
                    f"\nLoading dataset: Config='{config_name}', Split='{source_split}'"
                )

                dataset = load_dataset(
                    dataset_path,
                    name=config_name,
                    split=source_split,
                    streaming=True,
                )
                print(f"Successfully connected to stream for {config_name}")

                for entry in tqdm(
                    dataset,
                    total=max_documents_per_subset,
                    desc=f"Working on {config_name}",
                ):
                    if count >= max_documents_per_subset:
                        break

                    text = entry.get("text", "").strip()
                    if text:
                        if len(text) < 500: # ignore short fragments
                            continue

                        # ignore fragments with too much 'noise'
                        if len(re.findall(r"[\{\}\\\$\_\^\[\]\|]", text)) > 5:
                            continue

                        # ignore trash
                        if (
                            "mehr zum thema" in text.lower()
                            or "teilen auf twitter" in text.lower()
                        ):
                            continue

                        f.write(text + "\n" + end_of_text_token + "\n\n")
                        count += 1

                successfully_loaded.append(f"{config_name} ({source_split})")

            except Exception as e:
                print(f"Error in {config_name}: {e} - skipping..")

    print(
        f"\nSuccessfully downloaded the clean german corpus :) -> saved in {output_file}."
    )
    print(f"Managed to load: {', '.join(successfully_loaded)}")


if __name__ == "__main__":
    export_premium_german_corpus(max_documents_per_subset=1500)
