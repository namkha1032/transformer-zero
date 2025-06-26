import torch
from torch.nn import functional as F
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)

# Define dimensions
T = 5  # Time steps
E = 5  # Embedding size
tril = torch.tril(torch.ones(T, T))

# Create tensor with random numbers
wei = torch.rand(T, E)
wei = wei.masked_fill(tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
wei = F.softmax(wei, dim=-1)  # (B,T,T)


dropout = nn.Dropout(0.5)

wei = dropout(wei)

print(wei)