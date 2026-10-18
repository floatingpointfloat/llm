"""
Create vocabulary using the BPE Byte Level Tokenizer
"""

from pathlib import Path

from tokenizer import BPETokenizer
from config import training_file, tokenizer_file

# path to the training corpus
training_file = Path("data\physics_training_corpus.txt")

# initialize the tokenizer
tokenizer = BPETokenizer(vocab_size=20000)

# train the tokenizer
tokenizer.train([str(training_file)])

# save the tokenizer
tokenizer.save(tokenizer_file)

print("Tokenizer training completed.")
print("Vocabulary size:", tokenizer.get_vocab_size())
