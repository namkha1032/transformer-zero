import torch

from hyperparams import device
from transformer_zero import TransformerZeroModel

model = TransformerZeroModel().to(device)
print(model)
pass