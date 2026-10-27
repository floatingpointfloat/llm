import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

print("=== Einfaches Hugging Face Text-Completion ===")

# Kleines Modell laden (ca. 200MB)
model_name = "gpt2"  # GPT-2 klein - kostenlos und einfach

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name).cuda()
        if torch.cuda.is_available()
        else AutoModelForCausalLM.from_pretrained(model_name)
    )

    # Modell eval-Modus (kein Training!)
    model.eval()

    print(f"Modell '{model_name}' geladen.")
    print("Gib einen Text ein, der vervollständigt werden soll:")
    print("- 'exit' oder Ctrl+D zum Beenden")
    print("- 'quit' um zu beenden")
    print()

    while True:
        try:
            user_input = input("Du: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Auf Wiedersehen!")
                break

            # Tokenisieren und Padding
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True)

            # Generieren (10 neue Tokens)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50, temperature=0.7)

            # In Text konvertieren
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Nur den neuen Teil anzeigen (Original + Generation)
            original_length = len(user_input.replace(" ", ""))
            completion_start = (
                generated_text.find(user_input, 10)
                if user_input in generated_text
                else -1
            )

            if completion_start > 0:
                print(
                    f"\nModell: {generated_text[completion_start+max(completion_start, 1):]}"
                )
            else:
                print(f"\nModell:\n{generated_text}")

        except Exception as e:
            print(f"Error: {e}")

except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    print("Stelle sicher, dass PyTorch und transformers installiert sind:")
    print("pip install torch transformers")
