from datasets import load_dataset
from tqdm import tqdm



def export_german_corpus(output_file="training_text.txt", dataset_path="coral-nlp/german-commons"): # path to a german corpus dataset
    print("Starting the download of the german corpus...")
    
    #loading the corpus
    dataset = load_dataset(dataset_path, split="train", streaming=True)
    
    end_of_text_token = "<|endoftext|>"
    
    print("Writing file...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, entry in enumerate(tqdm(dataset, desc="Working on documents")):
            text = entry.get("text", "").strip()
            
            if text:
                f.write(text + "\n" + end_of_text_token + "\n")
                
    print(f"Successfully downloaded the german corpus :) - saved in {output_file}.")
    
if __name__ == "__main__":
    export_german_corpus()