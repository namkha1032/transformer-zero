import torch

batch_size = 64  # how many independent sequences will we process in parallel?
context_size = 256  # what is the maximum context length for predictions?
n_embed = 384
n_layer = 6
n_head = 6
epochs = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
dropout = 0.2


# batch_size = 32 # how many independent sequences will we process in parallel?
# block_size = 8 # what is the maximum context length for predictions?
# n_embed = 32
# n_layer = 3
# n_head = 4
# epochs = 10
# learning_rate = 1e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# dropout = 0.2
