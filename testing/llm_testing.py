import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Voraussetzungen prüfen
print(f"🔍 GPU verfügbar: {torch.cuda.is_available()}")

# 2. Modell laden
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # float16 spart RAM, aber float32 ist stabiler
    device_map="auto",  # automatisch auf GPU/CPU verteilen
)
print(f"✅ Modell geladen: {model.config.num_hidden_layers} Schichten")


# 3. Inferenz-Funktion
def generate(prompt, max_new_tokens=50):
    encoded = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        output = model.generate(encoded, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# 4. Test-Fragen
test_questions = [
    "Wer ist ein bekannter deutscher Autor?",
    "Was bedeutet KI?",
]

for q in test_questions:
    print(f"\n📝 Frage: {q}")
    print(f"🤖 Antwort: {generate(q, max_new_tokens=30)}")
