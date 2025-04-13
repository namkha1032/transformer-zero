import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ShakespeareDataset(Dataset):
    def __init__(self, file_path, seq_length=8):
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Get unique characters
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create character-to-index and index-to-character mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Convert text to list of indices
        self.data = [self.char_to_idx[ch] for ch in text]
        
        # Sequence length (context size)
        self.seq_length = seq_length

    def __len__(self):
        # Number of possible sequences
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Get a sequence of indices (input) and the next character indices (target)
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )


# Initialize dataset
dataset = ShakespeareDataset('input.txt', seq_length=8)

# Create DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(len(dataset))
# Test the DataLoader
for input_batch, target_batch in dataloader:
    print(f"Input batch shape: {input_batch.shape}")  # [batch_size, seq_length]
    print(f"Target batch shape: {target_batch.shape}")  # [batch_size, seq_length]
    print(f"Input batch shape: {input_batch}")  # [batch_size, seq_length]
    print(f"Target batch shape: {target_batch}")  # [batch_size, seq_length]
    break  # Just print one batch for verification