from hyperparams import batch_size
import torch
from tqdm import tqdm

def reshape(pred, y):
    BT, C = pred.shape
    new_y = y.view(BT)
    return new_y
    

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    pbar = tqdm(dataloader)
    for batch, (X, y) in enumerate(pbar):
        # Compute prediction and loss
        pred = model(X)
        # namkha: convert loss dim from (B, T, C) to (B*T, C)
        new_y = reshape(pred, y)
        loss = loss_fn(pred, new_y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print
        pbar.set_description(f"Loss: {loss.item():.4f}")
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    # sample_size = len(dataloader.dataset)
    sample_size = 0
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            new_y = reshape(pred, y)
            test_loss += loss_fn(pred, new_y).item()
            correct += (pred.argmax(1) == new_y).type(torch.int).sum().item()
            sample_size += new_y.shape[0]

    test_loss /= num_batches
    correct /= sample_size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")