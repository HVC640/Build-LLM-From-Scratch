from torch import nn
import torch


class PositionalEmbedding(nn.Module):
    def __init__(self, context_length, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
