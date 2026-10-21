from datasets import load_dataset
from tqdm import tqdm
import re


def is_clean_german_text(text):
    dutch_words = [" van ", " een ", " het ", " voor ", " zijn ", " niet "]
    if any(word in text.lower() for word in dutch_words):
        return False

    total_chars = len(text)
    digit_chars = len(re.findall(r"\d", text))
    if (digit_chars / total_chars) > 0.06:
        return False

    english_words = [" of ", " and ", " the ", " volume ", " index "]
    english_count = sum(text.lower().count(w) for w in english_words)
    if english_count > 3:
        return False

    if len(re.findall(r"[\(\)\[\]\.\-\:]", text)) > (total_chars * 0.02):
        return False

    return True


def export_german_corpus(
    output_file="training_text.txt",
    dataset_path="coral-nlp/german-commons",
    max_documents_per_subset=1000,
):
    print("Starting the download of the german corpus...")

    end_of_text_token = "<|endoftext|>"
    targets = [("cultural", "wikisource"), ("cultural", "blbooks")]
    loaded_subsets = []

    print("Writing file...")

    with open(output_file, "w", encoding="utf-8") as f:
        for config_name, source_split in targets:
            try:
                count = 0

                print(
                    f"Loading dataset for subset {config_name} ({source_split})"
                )
                dataset = load_dataset(
                    dataset_path,
                    name=config_name,
                    split=source_split,
                    streaming=True,
                )
                print(
                    f"Successfully loaded dataset for subset {config_name} ({source_split})"
                )

                for entry in tqdm(
                    dataset,
                    total=max_documents_per_subset,
                    desc=f"Working on document {source_split}",
                ):
                    if count >= max_documents_per_subset:
                        break

                    text = entry.get("text", "").strip()
                    if text:
                        if len(text) < 600:
                            continue

                        if (
                            len(re.findall(r"[\{\}\\\$\_\^\[\]\|]", text)) > 10
                        ):
                            continue

                        if not is_clean_german_text(text):
                            continue

                        text = text.replace("huchstens", "höchstens")
                        text = text.replace("Mahrchen", "Märchen")
                        text = text.replace("Erzahlungen", "Erzählungen")

                        f.write(text + "\n" + end_of_text_token + "\n\n")
                        count += 1

                loaded_subsets.append(f"{config_name} ({source_split})")

            except Exception as e:
                print(f"Error in subset {source_split} - skipping..")

    print(
        f"Successfully downloaded the german corpus :) - saved in {output_file}.\nManaged to load {len(loaded_subsets)} out of the subsets {len(targets)}"
    )


if __name__ == "__main__":
    export_german_corpus(max_documents_per_subset=1500)
