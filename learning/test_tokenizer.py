"""
Check the tokenizer for existing vocab
"""

from tokenizer import BPETokenizer

# load the trained tokenizer
tokenizer = BPETokenizer.load("tokenizer.json")

text = "Hello! I am building my own language model."

# encode text
token_ids = tokenizer.encode(text)

# decode token ids
decoded_text = tokenizer.decode(token_ids)

print("Vocabulary size:", tokenizer.get_vocab_size())
print("Token IDs:", token_ids)
print("Decoded text:", decoded_text)

assert decoded_text == text

"""Tests"""
text = "Quantum computers are fascinating! 12345 € 🚀"

token_ids = tokenizer.encode(text)
decoded_text = tokenizer.decode(token_ids)

assert decoded_text == text
print(decoded_text)
print("worked")
