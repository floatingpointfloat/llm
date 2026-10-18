import os
import re
import urllib.request

# Configuration
SOURCE_URL = "https://archive.org/stream/in.ernet.dli.2015.141932/2015.141932.Fundamental-University-Physics-Volume-1-Mechanics_djvu.txt"
OUTPUT_FILE = "physics_training_corpus.txt"
TARGET_SENTENCES = 3000


def download_and_build_dataset():
    print("Step 1: Downloading raw university physics text dump...")
    try:
        # Pretend to be a browser agent to prevent the host from blocking the automated download
        req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw_text = response.read().decode("utf-8")
    except Exception as e:
        print(
            f"[-] Execution Failed: Could not fetch data from the server. Details: {e}"
        )
        return

    print("Step 2: Processing, tokenizing, and cleaning sentence blocks...")
    # Normalize all spacing, tabs, and line breaks into uniform spaces
    cleaned_text = re.sub(r"\s+", " ", raw_text)

    # Advanced Regex to parse sentences accurately (ignores common abbreviations like i.e., e.g., Fig.)
    sentence_end_pattern = re.compile(
        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<!Fig\.)(?<=\.|\?|\!)\s"
    )
    all_sentences = sentence_end_pattern.split(cleaned_text)

    valid_sentences = []
    for sentence in all_sentences:
        sentence = sentence.strip()

        # Rigorous heuristic filtering tailored for AI training data:
        # 1. Length constraint (between 40 and 250 characters) to remove mathematical equation fragments or raw page headers.
        # 2. Case checking: Must start with a capital letter to ensure it's a complete sentence.
        # 3. Quality rule: Excludes math artifacts that contain sequential equals or slash markers.
        if (
            len(sentence) >= 40
            and len(sentence) <= 250
            and re.match(r"^[A-Z]", sentence)
            and not sentence.count("=") > 1
            and not sentence.startswith("PAGE")
        ):

            valid_sentences.append(sentence)

    print(f"[+] Total structural sentences discovered: {len(valid_sentences)}")

    # Slice exactly down to the target 3,000 sentences
    if len(valid_sentences) >= TARGET_SENTENCES:
        final_corpus = valid_sentences[:TARGET_SENTENCES]
        print(f"[+] Extracted exactly {TARGET_SENTENCES} premium sentences.")
    else:
        final_corpus = valid_sentences
        print(
            f"[!] Warning: Only managed to extract {len(valid_sentences)} matching sentences."
        )

    print(f"Step 3: Exporting data to plain text file...")
    # Write directly to your local file system as a raw newline-delimited file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sentence in final_corpus:
            f.write(sentence + "\n")

    print("\n" + "=" * 50)
    print(" SUCCESSFUL EXECUTION COMPLETE")
    print(f" File Location: {os.path.abspath(OUTPUT_FILE)}")
    print(f" File Format: Plain UTF-8 Text (.txt)")
    print(f" Line Count: {len(final_corpus)} sentences")
    print("=" * 50)


if __name__ == "__main__":
    download_and_build_dataset()
