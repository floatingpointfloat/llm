"""
Universal BPE Byte Level Tokenizer using libraries.
Did a BPE Tokenizer myself to pretty much fully understand how it works - this one needs to be good tho lol
"""

from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders


class BPETokenizer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size

        # initialize a byte-level bpe tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token=None))

        # split text into byte-level units before bpe
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # convert tokens back into readable text
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, files):
        # initialize the bpe trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # train the vocabulary and merge rules
        self.tokenizer.train(files, trainer)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def save(self, path):
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path):
        instance = cls.__new__(cls)
        instance.tokenizer = Tokenizer.from_file(str(path))
        instance.vocab_size = instance.tokenizer.get_vocab_size()
        return instance

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
