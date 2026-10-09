import re

text = "This is a example. I like neural networks!"


def tokenize(text):
    tokens = re.findall(
        r"\w+|[^\w\s]", text
    )  # finding every single sign / word using regular expressions

    return tokens


def build_vocab(text):
    tokens = tokenize(text)
    # create vocab
    vocab = sorted(set(tokens) | {"<UNK>"})  # unknown things as well
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    return token_to_id, id_to_token


def encode(text, token_to_id):
    tokens = tokenize(text)
    unk_id = token_to_id["<UNK>"]
    token_ids = [token_to_id.get(token, unk_id) for token in tokens]

    return token_ids


def decode(token_ids, id_to_token):
    tokens = [id_to_token[token_id] for token_id in token_ids]
    text = " ".join(tokens)

    # Remove spaces before common punctuation marks
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)  # more regular expression stuff

    return text


training_text = "I like cats and dogs."

token_to_id, id_to_token = build_vocab(training_text)

new_text = "I like birds."

token_ids = encode(new_text, token_to_id)

print("Vokabular:", token_to_id)
print("Neuer Text:", new_text)
print("Token-IDs:", token_ids)
print("Dekodiert:", decode(token_ids, id_to_token))
