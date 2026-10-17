"""
'GPT' model - decoder only architecture
"""

from dataclasses import dataclass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# dataclass that saves the config of the model
# choose the dataclass decorator cause its nice for storing data :)
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128  # sequence length
    n_embd: int = 256  # size of the vector per token
    n_heads: int = 4  # amount of heads for the multihead attention
    n_layers: int = 4
    dropout: float = 0.1


# multihead self attention based on 'attention is all you need' from the Google Brain Team
# causal cause it can't look into future tokens :)
# all the used functions are from the paper (link: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.n_heads = config.n_heads
        assert (
            config.n_embd % self.n_heads == 0
        )  # can the dimensions be splitted equally across the heads? check the config if that's not possible
        self.head_dim = config.n_embd // self.n_heads
        """
        dimension per head: example 256 dimensions 4 head -> head_dim 64 dimensions per head
        integer division cause we sadly do not want floats here 
        """
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size))
        )  # create the causal mask in this form (more or less):
        # tensor([[1., 0., 0., 0.],
        # [1., 1., 0., 0.],
        # [1., 1., 1., 0.],
        # [1., 1., 1., 1.]])
        # prevent the model from looking at future tokens

        self.key = nn.Linear(
            config.n_embd, config.n_embd
        )  # the key layer, takes the entire token vector
        self.value = nn.Linear(
            config.n_embd, config.n_embd
        )  # the value layer, also takes the entire token vector
        self.query = nn.Linear(
            config.n_embd, config.n_embd
        )  # the query layer, also takes the entire token vector

        # projection layer to combine the different vectors q, k, v
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)  # dropout layers
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor
    ):  # input tensor in the shape (batch_size, sequence_length, n_embd), for example (2, 10, 256)
        # go through the three attention methods
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # reshaping the value / query / key head into multihead attentions thingys
        # distribute the vectors across the heads (letting them focus on separate things)
        # afterwards it should have this form (in the following lines)
        # (batch_size, sequence_length, n_heads, head_dim)
        # example: (2, 10, 4, 64)
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.head_dim)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.head_dim)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.head_dim)

        # the dimensions have to be changed again so pytorch can do its thing
        # --> this (batch_size, sequence_length, n_heads, head_dim) to this (batch_size, n_heads, sequence_length, head_dim)
        # u can use the torch.transpose function
        # transpose(1, 2) change the positions of the first and second index (well, second and third lol)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # and now the key vector from (batch_size, n_heads, sequence_length, head_dim) to (batch_size, n_heads, head_dim, sequence_length)
        k = k.transpose(2, 3)

        # attention scores or part of them (page 4):
        scores = torch.matmul(q, k) / math.sqrt(self.head_dim)

        # apply causal masking to prevent the model from looking at future tokens
        sequence_length = q.shape[2]  # get the current sequence length from q

        # select the part of the mask that matches the current sequence length
        mask = self.mask[:sequence_length, :sequence_length]

        # replace scores at forbidden positions with negative infinity
        # this ensures that these positions receive zero attention after softmax
        scores = scores.masked_fill(mask == 0, -torch.inf)

        # convert the attention scores into attention weights
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # combine the value vectors using the attention weights
        out = torch.matmul(weights, v)
        # move the sequence length dimension before the heads
        # (batch_size, n_heads, sequence_length, head_dim) to (batch_size, sequence_length, n_heads, head_dim)
        out = out.transpose(1, 2)
        # combine the vectors
        out = out.reshape(out.shape[0], out.shape[1], self.config.n_embd)

        # projecting the ouput with the learned projection layer (explanation above in the initialization)
        out = self.proj(out)
        out = self.resid_dropout(out)

        return out


# feed forward part of the network
# individual 'work' on a singular token
# expanding between the layers to give the model 'room' to breathe
class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.ff = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor):
        out = self.ff(x)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor):
        # attention block
        out = self.ln1(x)
        out = self.attention(out)
        out = out + x
        residual_point = out

        # feed-forward block
        out = self.ln2(out)
        out = self.ff(out)
        out = out + residual_point

        return out


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # token embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.n_embd
        )
        # positional embedding
        self.positional_embedding = nn.Embedding(
            num_embeddings=config.block_size, embedding_dim=config.n_embd
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )  # transformer blocks :) stored in a list basically

        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size
        )  # language model head

    def forward(self, idx: torch.Tensor):
        # get the sequence length
        sequence_length = idx.shape[1]

        if (
            sequence_length > self.config.block_size
        ):  # sequence and block size don't match
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds "
                f"block size ({self.config.block_size})"
            )

        # create position indices
        positions = torch.arange(sequence_length, device=idx.device)

        # calculate embeddings
        token_emb = self.token_embedding(idx)
        pos_emb = self.positional_embedding(positions)

        # combine token and positional embeddings
        x = token_emb + pos_emb

        for block in self.blocks:  # iterate through the transformer blocks
            x = block(x)

        x = self.ln_f(x)  # final layer norm
        x = self.lm_head(x)  # language model head

        return x


"""
Debug :))
"""

config = GPTConfig(vocab_size=100)
model = GPT(config)

idx = torch.randint(0, config.vocab_size, (2, 10))
logits = model(idx)

print(logits.shape)


model.train()

logits_1 = model(idx)
logits_2 = model(idx)

print("Training mode:", not torch.equal(logits_1, logits_2))

model.eval()

logits_3 = model(idx)
logits_4 = model(idx)

print("Evaluation mode:", torch.equal(logits_3, logits_4))
