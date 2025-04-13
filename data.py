import torch
from torch.utils.data import Dataset, DataLoader
from hyperparams import batch_size, block_size, device
from cipher import encode

class TextDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        self.stoi:dict = None
        self.itos:dict = None
        self.data:list = None
        self.transform = transform
        self.target_transform = target_transform
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        unique_chars = sorted(list(set(text)))
        
        self.stoi = { ch:i for i,ch in enumerate(unique_chars) }
        self.itos = { i:ch for i,ch in enumerate(unique_chars) }
        self.data = torch.tensor(encode(text, self.stoi), dtype=torch.long).to(device)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+block_size].to(device)
        y = self.data[idx+1:idx+block_size+1].to(device)
        # x, y = x.to(device), y.to(device)
        return x, y
    
    
if __name__ == "__main__":
    dataset = TextDataset('./input.txt')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(len(dataset))
    # Test the DataLoader
    for input_batch, target_batch in dataloader:
        print(f"Input batch shape: {input_batch.shape}")  # [batch_size, seq_length]
        print(f"Target batch shape: {target_batch.shape}")  # [batch_size, seq_length]
        print(f"Input batch shape: {input_batch}")  # [batch_size, seq_length]
        print(f"Target batch shape: {target_batch}")  # [batch_size, seq_length]
        break  # Just print one batch for verification