import torch

from data.vocab import itos, stoi
from func.cipher import decode, encode
from hyperparams import device
from transformer_zero import TransformerZeroModel

model = TransformerZeroModel().to(device)
model.load_state_dict(torch.load("zero.pth", map_location=device))
model.eval()


print("device is: ", device)
# # count
# def count_learnable_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# count = count_learnable_params(model)
# pass


# generate from the model
init_text = """Sunny hills bloom with vivid colors as ants march in neat rows to gather tiny bits of sweet nectar. Each step, a dance of purpose, weaves a tale of unity. Day fades, stars peek, and the cycle spins on. Numbers tick, guiding their path. Life hums in small acts, a quiet, endless song."""
# new_text = init_text[-256:]
new_text = """d colors as ants march in neat rows to gather tiny bits of sweet nectar. Each step, a dance of purpose, weaves a tale of unity. Day fades, stars peek, and the cycle spins on. Numbers tick, guiding their path. Life hums in small acts, a quiet, endless song?"""
init_embed = torch.tensor([encode(new_text, stoi)], dtype=torch.long, device=device)
# abc = torch.tensor([[0,2,5,4,3]]).to(device)
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(init_embed, max_new_tokens=500)[0].tolist(), itos))
