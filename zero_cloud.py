import torch
from transformer_zero import TransformerZeroModel
from hyperparams import device
from cipher import decode
from vocab import itos

model = TransformerZeroModel().to(device)
model.load_state_dict(torch.load("zero.pth"))
model.eval()

# generate from the model
abc = torch.tensor([[0,2,5,4,3]]).to(device)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(abc, max_new_tokens=500)[0].tolist(), itos))
