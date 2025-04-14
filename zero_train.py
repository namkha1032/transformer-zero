import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import math
from transformer_zero import TransformerZeroModel
from hyperparams import device, learning_rate, epochs, batch_size
from train_test import train_loop, test_loop
from dataclass import TextDataset
from vocab import data

def split_dataset(dataset: Dataset, test_ratio: float = 0.1):
    total_size = len(dataset)
    test_size = math.floor(total_size * test_ratio)
    train_size = total_size - test_size
    
    # Indices for test (first 10%) and train (last 90%)
    test_indices = list(range(0, test_size))
    train_indices = list(range(test_size, total_size))
    
    test_dataset = Subset(dataset, test_indices)
    train_dataset = Subset(dataset, train_indices)
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    model = TransformerZeroModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    dataset = TextDataset(data)
    # Train test split
    generator1 = torch.Generator().manual_seed(42)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)
    # train_dataset, test_dataset = split_dataset(dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size)
    train_dataloader = DataLoader(dataset, batch_size)
    # test_dataloader = DataLoader(test_dataset, batch_size)
    
    
    
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        test_dataloader = DataLoader(test_dataset, batch_size)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    
    # save model

    torch.save(model.state_dict(), "zero.pth")
    print("Saved PyTorch Model State to zero.pth")

        