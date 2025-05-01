def encode(text_seq, stoi):
    return [stoi[c] for c in text_seq]

def decode(num_seq, itos):
    return ''.join([itos[i] for i in num_seq])