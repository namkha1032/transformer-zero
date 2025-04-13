import torch
import torch.nn as nn
from torch.nn import functional as F
from data.vocab import vocab
from data.dataclass import TextDataset
from hyperparams import n_embed, block_size, device

# Create vocabulary
vocab_size = len(vocab)


class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # This is typically used to register a buffer that should not to be considered a model parameter
        # Buffers, by default, are persistent and will be saved alongside parameters.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        
        # compute attention scores ("affinities")
        # scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) = (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        
        # perform weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) = (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # Run all heads parallel and concat all outputs over the channel (C) dimension
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
class FeedForward(nn.Module):
    """simple linear layer foloowed by a non-linearity"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

    

class TransformerZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = MultiHeadAttention(4, n_embed//4)
        # The tokens look at each other but does not have time to think on what they found from others
        # This feedforward is the per token level --> they already communicate and gather data in the self attention layer, now they need to think on that data individually
        # That's the purpose of the feed forward layer
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        # idx, targets: (B,T)
        token_embed = self.token_embedding_table(idx) # (B,T,C) --> C is n_embed (embed dimension)
        position_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        # Broadcasting (B,T,C) + (T,C)
        x = token_embed + position_embed # (B,T,C)
        # Apply one head of self-attention (B,T,C)
        x = self.sa_head(x)
        # Apply feed forward
        x = self.ffwd(x)
        # logits: (B,T,vocab_size)
        logits = self.lm_head(x)
        B, T, C = logits.shape            
        # we have to do this because cross_entropy expect (B,C,T), not (B,T,C)
        logits = logits.view(B*T, C)
        return logits
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # We have to make sure idx is never more than block_size, otherwise, position_embedding_table will run out of scope
            # That's why we crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            BT, C = logits.shape
            logits = logits[-1, :].reshape(1, C) # becomes (1,65)
            # logits = torch.reshape(logits, (65,))
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

    
