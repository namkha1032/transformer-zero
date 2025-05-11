import torch
from torch.utils.data import DataLoader, Dataset

from func.cipher import encode
from hyperparams import batch_size, context_size, device


class TextDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data: list = torch.tensor(data).to(device)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # Number of possible sequences
        # return len(self.data) - context_size
        return len(self.data) // context_size

    def __getitem__(self, idx):
        context_idx = idx * context_size
        x = self.data[context_idx : context_idx + context_size].to(device)
        y = self.data[context_idx + 1 : context_idx + 1 + context_size].to(device)
        # x, y = x.to(device), y.to(device)
        return x, y


if __name__ == "__main__":
    dataset = TextDataset("./input.txt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(len(dataset))
    # Test the DataLoader
    for input_batch, target_batch in dataloader:
        print(f"Input batch shape: {input_batch.shape}")  # [batch_size, seq_length]
        print(f"Target batch shape: {target_batch.shape}")  # [batch_size, seq_length]
        print(f"Input batch shape: {input_batch}")  # [batch_size, seq_length]
        print(f"Target batch shape: {target_batch}")  # [batch_size, seq_length]
        break  # Just print one batch for verification
