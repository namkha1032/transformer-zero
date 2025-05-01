import torch
import torch.nn as nn
from torch.nn import functional as F

from data.vocab import vocab_size
from hyperparams import block_size, device, dropout, n_embed, n_head, n_layer

# Create vocabulary
# vocab_size = len(vocab)


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # This is typically used to register a buffer that should not to be considered a model parameter
        # Buffers, by default, are persistent and will be saved alongside parameters.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # drop out
        #   - at train time: randomly drop neuron and train without them (change every forward backward pass) --> train on an ensemble of sub networks
        #   - at test time: everything is fully enabled --> all of sub-network are merged into a single one
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)

        # compute attention scores ("affinities")
        # scaled dot-product attention
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B,T,hs) @ (B,hs,T) = (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        # Drop out the affinity to randomly prevent some of the node to communicate
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) = (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection layer going back to the residual pathway? the Wo in the paper?
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads parallel and concat all outputs over the channel (C) dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """simple linear layer foloowed by a non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # Growing this layer in the residual block on the side of the residual pathway
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # projection? (same as in multihead)
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer blockL: communication (sa) followed by computation (ffwd)"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.ln1 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)
        # The tokens look at each other but does not have time to think on what they found from others
        # This feedforward is the per token level --> they already communicate and gather data in the self attention layer, now they need to think on that data individually
        # That's the purpose of the feed forward layer

    def forward(self, x):
        # Apply layernorm first (pre-norm)
        x = self.ln1(x)
        # Apply multi-head self-attention with residual connection
        x = x + self.sa(x)  # (B,T,C)
        # Apply layernorm first (pre-norm)
        x = self.ln2(x)
        # Apply feed forward with residual connection
        x = x + self.ffwd(x)  #
        return x


class TransformerZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.blocks = nn.Sequential(
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     # this is new
        #     nn.LayerNorm(n_embed)
        # )
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)]
        )
        # this is new
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        # idx, targets: (B,T)
        token_embed = self.token_embedding_table(idx)  # (B,T,C)
        position_embed = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,C)
        # Broadcasting (B,T,C) + (T,C)
        x = token_embed + position_embed  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        B, T, C = logits.shape
        # we have to do this because cross_entropy expect (B,C,T), not (B,T,C)
        logits = logits.view(B * T, C)
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
            logits = logits[-1, :].reshape(1, C)
            # logits = torch.reshape(logits, (65,))
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1) # (B,1), random
            idx_next = probs.argmax(dim=-1, keepdim=True)  # (B,1), no random
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx
