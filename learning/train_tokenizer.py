"""
Create vocabulary using the BPE Byte Level Tokenizer
"""

from pathlib import Path

from tokenizer import BPETokenizer
from config import training_file, tokenizer_file

print("Training file:", repr(training_file))
print("String path:", repr(str(training_file)))
print("File exists:", Path(training_file).exists())
print("Absolute path:", Path(training_file).resolve())

# initialize the tokenizer
tokenizer = BPETokenizer(vocab_size=8000)

# train the tokenizer
tokenizer.train([str(training_file)])

# save the tokenizer
tokenizer.save(tokenizer_file)

print("Tokenizer training completed.")
print("Vocabulary size:", tokenizer.get_vocab_size())
