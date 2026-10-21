import os

def check_training_corpus(file_path="training_text.txt"):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    documents = content.split("<|endoftext|>")
    doc_count = len(documents) - 1 

    print(f"--- Corpus Statistics ---")
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Extracted Documents: {doc_count}")
    
    if doc_count > 0:
        print(f"\n--- Preview of First Clean Document ---")
        print(documents[0].strip()[:1000] + "\n...")
        print(f"----------------------------------------")

if __name__ == "__main__":
    check_training_corpus()
