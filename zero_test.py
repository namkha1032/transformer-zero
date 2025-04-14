import torch
from transformer_zero import TransformerZeroModel
from hyperparams import device
from cipher import decode
from vocab import itos

model = TransformerZeroModel().to(device)
model.load_state_dict(torch.load("zero.pth"))
model.eval()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist(), itos))
