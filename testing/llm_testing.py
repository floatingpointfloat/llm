import torch
import torch.nn as nn
from collections import Counter
import re

class SimpleTextModel(nn.Module):
    """Ein sehr einfaches Text-Klassifizierungsmodell mit festen Gewichten (kein Training)"""
    
    def __init__(self, vocab_size=100, embedding_dim=8, hidden_dim=16):
        super().__init__()
        
        # Einfache feste Embeddings (zufällig initialisiert, aber fixiert)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Kleine verborgene Schicht mit festen Gewichten
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        
        # Ausgabeschicht für Klassifikation
        self.output = nn.Linear(hidden_dim, 3)  # 3 Klassen: positiv, negativ, neutral
        
        # Feste Initialisierung (kein Training)
        self.embedding.weight.data.zero_()
        self.hidden.weight.data = torch.randn(embedding_dim, hidden_dim) * 0.1
        self.output.weight.data = torch.randn(hidden_dim, 3) * 0.5
        self.hidden.bias.data.zero_()
        self.output.bias.data.zero_()
    
    def tokenize(self, text):
        """Einfache Tokenisierung: extrahiert nur englische Wörter"""
        words = re.findall(r'[a-zA-Z]+', text.lower())
        return words
    
    def predict(self, text):
        """Zieht Vorhersage basierend auf dem Text"""
        # Tokenisieren
        tokens = self.tokenize(text)
        
        if not tokens:
            return "Keine gültigen Wörter gefunden"
        
        # Einfacher Lookup: Hashing der Tokens zu Indices (super simpel!)
        def hash_word(word):
            # Einfacher Deterministischer Hash für jedes Wort
            return sum(ord(c) for c in word) % 100
        
        # Tokens in Indices konvertieren
        indices = [hash_word(token) for token in tokens]
        
        if not indices:
            return "Keine gültigen Wörter gefunden"
        
        # Forward Pass
        x = self.embedding(torch.LongTensor(indices))
        h = torch.relu(self.hidden(x))
        logits = self.output(h)
        probs = torch.softmax(logits, dim=1)
        
        # Klassifikation
        pred_class = torch.argmax(probs, dim=1).item()
        
        labels = ["neutral", "positiv", "negativ"]
        confidence = probs[0][pred_class].item() * 100
        
        return {
            'text': text,
            'tokens': tokens,
            'prediction': labels[pred_class],
            'confidence': f"{confidence:.2f}%",
            'all_probs': {labels[i]: f"{probs[0][i].item()*100:.2f}%" 
                         for i in range(3)}
        }


# Initialisiere das Modell (keine Gewichte trainieren)
model = SimpleTextModel()

print("=== Einfaches Text-Klassifizierungsmodell ===")
print("(Feste Gewichte, kein Training)")
print()

# Beispiel-Nutzerinput testen
user_input = "Ich liebe dieses Programm!"
result = model.predict(user_input)
print(f"Input: {user_input}")
print(f"Vorhersage: {result['prediction']}")
print(f"Kontext-Wahrscheinlichkeiten: {result['all_probs']}")
print()

# Mehrere Tests mit verschiedenen Inputs
test_cases = [
    "Das war schlecht",
    "Wunderschön!",
    "Alles okay vielleicht",
    "Ich hasse das nicht wirklich aber nicht auch"
]

print("--- Weitere Testfälle ---")
for test in test_cases:
    result = model.predict(test)
    print(f"\nInput: '{test}'")
    print(f"→ {result['prediction']}: {result['confidence']}")