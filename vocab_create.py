from cipher import encode, decode

if __name__ == "__main__":
    with open('./input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    with open('./vocab.py', 'w', encoding='utf-8') as f:
        vocab_line = f"vocab = {str(chars)}"
        stoi_line = f"stoi = {str({ ch:i for i,ch in enumerate(chars) })}"
        itos_line = f"itos = {str({ i:ch for i,ch in enumerate(chars) })}"
        data_line = f"data = {encode(text)}"
        
        f.write(f"{vocab_line}\n{stoi_line}\n{itos_line}\n{data_line}")

    pass
    pass