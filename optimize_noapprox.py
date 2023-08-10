import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.func import functional_call, hessian
from torch.autograd import grad
import random

random.seed(0)
torch.manual_seed(123)
np.random.seed(123)

def data_generating(n):
    xs = np.random.randn(5, n)
    y = 0.5*xs[0] + xs[0]**2 + np.exp(xs[1])/2 + (xs[2]*xs[3])/7 - np.log(abs(xs[3])) + np.sin(xs[4]) + np.random.randn(n)
    return xs.T, y

def compute_loss(params, mod, x, y, loss_fn, lambda_val=0):
    yhat = functional_call(mod, params, x)
    reg_term = torch.norm(torch.cat([param.view(-1) for param in mod.parameters()]), p=2) ** 2
    return loss_fn(yhat, y) + 0.5*lambda_val*reg_term


n = 10000
val_n = 20000
x_train, y_train = data_generating(n)
x_train = torch.as_tensor(x_train).float()
y_train = torch.as_tensor(y_train).float().view(n, -1)
train_dat = TensorDataset(x_train, y_train)

x_val, y_val = data_generating(val_n)
x_val = torch.as_tensor(x_val).float()
y_val = torch.as_tensor(y_val).float().view(val_n, -1)
val_dat = TensorDataset(x_val, y_val)

batch_size = 64
train_loader = DataLoader(dataset=train_dat,
                          shuffle = True,
                          batch_size=batch_size)
val_loader = DataLoader(dataset=val_dat,
                          shuffle = True,
                          batch_size=batch_size)

lr = 0.01
beta = 0.01
min_delta = 0.0001
patience = 10
best_loss = float('inf')
counter = 0

model = nn.Sequential(nn.Linear(5, 4),
                      nn.ReLU(),
                      nn.Linear(4, 3),
                      nn.ReLU(),
                      nn.Linear(3, 2),
                      nn.ReLU(),
                      nn.Linear(2, 1))
num_params = [p.numel() for p in model.parameters()]
loss_fn = nn.MSELoss()

lambda_penalty = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))

# Training loop
best_loss = float('inf')
counter = 0
lambda_traj = []
val_losses = []
num_params = sum(p.numel() for p in model.parameters())
num_epochs = 5
for epoch in range(num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lambda_penalty.item())
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

    # Calculate average losses
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Check for improvement
    if val_loss + min_delta < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1

    # Check if early stopping criteria are met
    if counter >= patience:
        print("Early stopping: No improvement in validation loss.")
        break

    optimizer.zero_grad()
    loss_train = compute_loss(params=dict(model.named_parameters()), mod=model, x=x_train, y=y_train, loss_fn=loss_fn,
                              lambda_val=lambda_penalty)
    loss_val = compute_loss(params=dict(model.named_parameters()), mod=model, x=x_val, y=y_val, loss_fn=loss_fn,
                            lambda_val=0)

    dlv_dw = grad(loss_val, model.parameters(), create_graph=True)
    dlv_dw = torch.cat([grad.view(-1) for grad in dlv_dw])

    dlt_dw = grad(loss_train, model.parameters(), create_graph=True)
    dlt_dw = torch.cat([x.flatten() for x in dlt_dw])

    # dlt_dlamda = grad(loss_train, lambda_penalty,  create_graph=True)[0]
    # dlt_dlamda = torch.cat([x.flatten() for x in dlt_dw])
    d2lt_dw2 = hessian(compute_loss, argnums=0)(dict(model.named_parameters()), model, x_train, y_train, loss_fn,
                                                lambda_penalty)
    H_nn = []
    with torch.no_grad():
        for i in d2lt_dw2:
            for h in d2lt_dw2[i]:
                H_nn.append(d2lt_dw2[i][h].flatten())
        H_nn = torch.cat(H_nn)
        H_nn = H_nn.reshape((num_params, num_params))
        d2lt_dw2_inv = torch.linalg.inv(H_nn)
        p = dlv_dw @ d2lt_dw2_inv
    # dlv_dlambda = grad(loss_val, lambda_penalty)[0]
    # print(dlv_dlambda)
    # print(dlt_dw)
    v3 = grad(dlt_dw, lambda_penalty, grad_outputs=p, retain_graph=True)[0]
    print(v3)
    with torch.no_grad():
        newval = lambda_penalty.item() - beta * (0 - v3)
        lambda_penalty.copy_(newval)
        lambda_penalty.data.clamp_(min=0, max=2)
        lambda_traj.append(lambda_penalty.item())


