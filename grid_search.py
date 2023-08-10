import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt


random.seed(0)
torch.manual_seed(123)

def data_generating(n):
    xs = np.random.randn(5, n)
    y = 0.5*xs[0] + xs[0]**2 + np.exp(xs[1])/2 + (xs[2]*xs[3])/7 - np.log(abs(xs[3])) + np.sin(xs[4]) + np.random.randn(n)
    return xs.T, y

def create_data_loader(n, batch_size):
    x, y = data_generating(n)
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float().view(n, -1)
    dat = TensorDataset(x, y)
    data_loader = DataLoader(dataset=dat, shuffle=True, batch_size=batch_size)
    return data_loader

def create_model():
    model = nn.Sequential(
        nn.Linear(5, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
        nn.ReLU(),
        nn.Linear(2, 1))
    return model

def train_model_with_lambda(model, train_loader, val_loader, lr, l, min_delta, patience, epochs):
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    counter = 0
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)  # save validation loss for each epoch

        if val_loss + min_delta < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at lambda = {l}: No improvement in validation loss.")
            break

    return val_losses

n = 10000
val_n = 20000
batch_size = 32
lr = 0.01
min_delta = 0.0001
patience = 5
lambdas = np.linspace(0.01, 2, 10)
num_epochs = 50

train_loader = create_data_loader(n, batch_size)
val_loader = create_data_loader(val_n, batch_size)

def plot_validation_losses(lambdas, all_val_losses):
    for i, val_losses in enumerate(all_val_losses):
        plt.plot(range(1, len(val_losses) + 1), val_losses, label=f"λ={lambdas[i]:.2f}")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss for each Lambda')
    plt.legend()
    plt.show()

all_val_losses = []
for l in lambdas:
    model = create_model()
    val_losses = train_model_with_lambda(model, train_loader, val_loader, lr, l, min_delta, patience, num_epochs)
    all_val_losses.append(val_losses)

plot_validation_losses(lambdas, all_val_losses)




