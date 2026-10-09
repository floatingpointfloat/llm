import torch
import torch.nn as nn

VOCAB_SIZE = 1000
EMBEDDING_DIM = 64

embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)

token_ids = torch.tensor([42, 17, 305])

vectors = embedding(token_ids)

print("Token-IDs:", token_ids.shape)
print("Embeddings:", vectors.shape)
print("Parameter:", sum(p.numel() for p in embedding.parameters()))
