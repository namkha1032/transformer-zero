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
text1 = "Hello I am a dragon slayer"
text2 = "Hello I am nguyen nam kha."
text = [text1, text2]
init_embed = torch.tensor([encode(txt, stoi) for txt in text], dtype=torch.long, device=device)
# abc = torch.tensor([[0,2,5,4,3]]).to(device)
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    result_token_list = model.generate(init_embed, max_new_tokens=300, random=True)
    result_txt_list = [decode(x.tolist(), itos) for x in result_token_list]
    for txt in result_txt_list:
        print(txt)
        print("--------------------------------------------")
    # print(decode(model.generate(init_embed, max_new_tokens=500)[0].tolist(), itos))
