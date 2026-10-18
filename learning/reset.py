import os
from config import tokenizer_file, checkpoint_file

# remove the files from training
os.remove(tokenizer_file)
os.remove(checkpoint_file)
