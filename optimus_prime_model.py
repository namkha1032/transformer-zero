import torch
import torch.nn as nn
from torch.nn import functional as F
from vocab import vocab

# Hyperparameters
n_embed = 32

# Create vocabulary
vocab_size = len(vocab)


class OptimusPrimeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        # idx, targets: (B, T)
        # token_embed: (B, T, C) --> C is n_embed (embed dimension)
        token_embed = self.token_embedding_table(idx)
        # logits: (B, T, vocab_size)
        logits = self.lm_head(token_embed)
        B, T, C = logits.shape            
        # we have to do this because cross_entropy expect B*C*T, not B*T*C
        logits = logits.view(B*T, C)
        return logits
        