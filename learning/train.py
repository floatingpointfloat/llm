"""
Train the GPT model.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model import GPT, GPTConfig
from prepare_data import GPTDataset, training_file, tokenizer_file
from tokenizer import BPETokenizer
