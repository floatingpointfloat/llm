import torch
import torch.nn as nn

VOCAB_SIZE = 7
EMBEDDING_DIM = 32

token_embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)

MAX_SEQ_LENGTH = 128

position_embedding = nn.Embedding(
    num_embeddings=MAX_SEQ_LENGTH, embedding_dim=EMBEDDING_DIM
)

# Example token IDs
token_ids = torch.tensor([2, 6, 4, 0], dtype=torch.long)
position_ids = torch.arange(len(token_ids))

token_vectors = token_embedding(token_ids)
position_vectors = position_embedding(position_ids)

x = token_vectors + position_vectors

ATTENTION_DIM = 32

query_layer = nn.Linear(EMBEDDING_DIM, ATTENTION_DIM)
key_layer = nn.Linear(EMBEDDING_DIM, ATTENTION_DIM)
value_layer = nn.Linear(EMBEDDING_DIM, ATTENTION_DIM)

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

print("Token vectors:", token_vectors.shape)
print("Position vectors:", position_vectors.shape)
print("Combined vectors:", x.shape)

print("Token embedding für Position 0:")
print(token_vectors[0])

print("\nPositions-Embedding für Position 0:")
print(position_vectors[0])

print("\nKombinierter Vektor für Position 0:")
print(x[0])
