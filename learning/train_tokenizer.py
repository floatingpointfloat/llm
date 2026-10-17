"""
Create vocabulary using the BPE Byte Level Tokenizer
"""

from pathlib import Path
from tokenizer import BPETokenizer

# path to the training corpus
training_file = Path("data/training.txt")

# initialize the tokenizer
tokenizer = BPETokenizer(vocab_size=8000)

# train the tokenizer
tokenizer.train([str(training_file)])

# save the tokenizer
tokenizer.save("tokenizer.json")

print("Tokenizer training completed.")
print("Vocabulary size:", tokenizer.get_vocab_size())
