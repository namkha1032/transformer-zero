import torch
import torch.nn as nn
from torch.nn import functional as F
from bigram import BigramLanguageModel, decode, vocab_size


device = 'cuda' if torch.cuda.is_available() else 'cpu'
new_model = BigramLanguageModel(vocab_size)

new_model.load_state_dict(torch.load("bigram.pth", weights_only=True))
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(new_model.generate(context, max_new_tokens=500)[0].tolist()))
print("--------")