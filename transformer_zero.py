import torch
import torch.nn as nn
from torch.nn import functional as F
from data.vocab import vocab
from data.dataclass import TextDataset
from hyperparams import n_embed, device, learning_rate, epochs

# Create vocabulary
vocab_size = len(vocab)


class TransformerZeroModel(nn.Module):
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
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    
