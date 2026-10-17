import re
from collections import Counter


class BPE_Tokenizer:
    def __init__(self):
        self.merge_rules = []
        self.split_words = {}
        self.token_to_id = {}
        self.id_to_token = {}

    # return the word and how many times it is found in the text
    # for example: low low lower -> Counter({'low': 2, 'lower': 1})
    def get_word_frequencies(self, text):
        # preserve whitespace and words
        segments = re.findall(r"\s+|[^\s]+", text)
        return Counter(segments)

    # split each word into separate characters and save those to this word
    # for example: low -> 'l' 'o' 'w' saved to 'low'
    def split_words_into_characters(self, word_frequencies):
        split_words = {}

        for word, frequency in word_frequencies.items():
            split_words[word] = {
                "characters": list(word),
                "frequency": frequency,
            }

        return split_words

    # count how the characters lie next to each other and in which frequencies
    def count_adjacent_pairs(self, split_words):
        pair_frequencies = Counter()

        for data in split_words.values():
            characters = data["characters"]
            frequency = data["frequency"]

            for i in range(len(characters) - 1):
                pair = (characters[i], characters[i + 1])
                pair_frequencies[pair] += frequency

        return pair_frequencies

    # merge every occurrence of a token pair into one token
    # for example: ['l', 'o', 'w'] -> ['lo', 'w']
    def merge_pair(self, split_words, pair):
        first, second = pair

        for data in split_words.values():
            tokens = data["characters"]
            merged_tokens = []

            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == first
                    and tokens[i + 1] == second
                ):
                    merged_tokens.append(first + second)
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1

            data["characters"] = merged_tokens

    # train the tokenizer by repeatedly merging the most common pair
    def train(self, text, merge_num):
        word_frequencies = self.get_word_frequencies(text)

        self.split_words = self.split_words_into_characters(word_frequencies)

        self.merge_rules = []

        for _ in range(merge_num):
            pair_frequencies = self.count_adjacent_pairs(self.split_words)

            if not pair_frequencies:
                break

            most_common_pair = pair_frequencies.most_common(1)[0][0]

            self.merge_pair(self.split_words, most_common_pair)
            self.merge_rules.append(most_common_pair)

            print(f"Merged {most_common_pair[0]} + {most_common_pair[1]}")

        return self.split_words

    # encode text into token ids
    def encode(self, text):
        segments = re.findall(r"\s+|[^\s]+", text)
        token_ids = []

        for segment in segments:
            tokens = list(segment)

            # apply the learned merge rules in training order
            for first, second in self.merge_rules:
                merged_tokens = []
                i = 0

                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and tokens[i] == first
                        and tokens[i + 1] == second
                    ):
                        merged_tokens.append(first + second)
                        i += 2
                    else:
                        merged_tokens.append(tokens[i])
                        i += 1

                tokens = merged_tokens

            # convert tokens into ids
            for token in tokens:
                if token not in self.token_to_id:
                    raise ValueError(
                        f"Unknown token: {token!r}. "
                        "The tokenizer may need more training data."
                    )

                token_ids.append(self.token_to_id[token])

        return token_ids

    # decode token ids back into text
    def decode(self, token_ids):
        tokens = []

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token id: {token_id}")

            tokens.append(self.id_to_token[token_id])

        return "".join(tokens)

    # build the vocabulary using the bpe encoder
    def build_vocab(self, text):
        # include every character from the training text
        tokens = set(text)
        tokens.update(["\n", "\t", "\r"])

        # include every token created by the merge rules
        for first, second in self.merge_rules:
            tokens.add(first + second)

        # assign each token a unique id
        tokens = sorted(tokens)

        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}


"""
Debug :))
"""

training_text = "low low low lower lower newest newest newest"

tokenizer = BPE_Tokenizer()

tokenizer.train(training_text, merge_num=10)
tokenizer.build_vocab(training_text)

# test encoding and decoding
text = "low newest lower"

token_ids = tokenizer.encode(text)
decoded_text = tokenizer.decode(token_ids)

print("\nVocabulary:")
print(tokenizer.token_to_id)

print("\nOriginal text:")
print(repr(text))

print("\nToken IDs:")
print(token_ids)

print("\nDecoded text:")
print(repr(decoded_text))

assert decoded_text == text

print("\nEncoding and decoding successful!")
