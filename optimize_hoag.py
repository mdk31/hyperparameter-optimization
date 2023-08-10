import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.func import functional_call, hessian
from torch.autograd import grad

# Constants
SEED = 123
n_train = 5000
n_val = 1000
batch_size = 32
lr = 0.01
l_lr = 0.01
min_delta = 0.0001
patience = 25
num_epochs = 2

# Functions
def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    # random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_loss(mod, x, y, loss_fn, lambda_val=0.0):
    yhat = mod(x)
    if lambda_val == 0.0:
        return loss_fn(yhat, y)
    else:
        reg_term = torch.norm(torch.cat([param.view(-1) for param in mod.parameters()]), p=2) ** 2
        return loss_fn(yhat, y) + 0.5 * torch.exp(lambda_val) * reg_term


def neumann_series(mod, v2, I=50):
    v = v2.detach().clone()
    p = v2.detach().clone()
    for i in range(I):
        tmp_v = torch.autograd.grad(v2, mod.parameters(), grad_outputs=v, retain_graph=True)
        tmp_v = torch.cat([grad.view(-1) for grad in tmp_v])
        v = v - tmp_v
        p = p + v
    return p

def conjugate_gradient(A, b, mod, tol, max_iter):
    x = torch.ones_like(b)
    r = b - grad_and_flat(A, mod.parameters(), x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(max_iter):
        Ap = torch.dot(p, grad_and_flat(A, mod.parameters(), p))
        alpha = rs_old/Ap
        x = x + alpha*p
        r = r -alpha*p
        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < tol:
            break
        beta = rs_new/rs_old
        p = r + beta*p
        rs_old = rs_new

    return x


def data_generating(n):
    xs = torch.randn((5, n))
    y = 0.5 * xs[0] + xs[0] ** 2 + torch.exp(xs[1]) / 2 + (xs[2] * xs[3]) / 7 - torch.log(torch.abs(xs[3])) + torch.sin(
        xs[4]) + torch.randn(n)
    return xs.T, y.view(n, -1)

def grad_and_flat(f, w, go):
    vjp = grad(f, w, grad_outputs=go, retain_graph=True)
    vjp = torch.cat([x.view(-1) for x in vjp])
    return vjp

set_seed(SEED)

# Generate the data
x_train, y_train = data_generating(n_train)
x_val, y_val = data_generating(n_val)

# Create datasets and dataloaders
train_dat = TensorDataset(x_train, y_train)
val_dat = TensorDataset(x_val, y_val)

train_loader = DataLoader(dataset=train_dat, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(dataset=val_dat, shuffle=True, batch_size=batch_size)

# Model creation
model = nn.Sequential(nn.Linear(5, 4),
                      nn.ReLU(),
                      nn.Linear(4, 2),
                      nn.ReLU(),
                      nn.Linear(2, 1))

loss_fn = nn.MSELoss()

lambdas = np.linspace(0.01, 2, 10)
lambda_penalty = torch.nn.Parameter(torch.tensor(0.23111111111111113, requires_grad=True))

best_loss = float('inf')
counter = 0
lambda_traj = []
val_losses = []
num_params = sum(p.numel() for p in model.parameters())

for epoch in range(num_epochs):
    optimizer = optim.Adam(model.parameters(), weight_decay=np.exp(lambda_penalty.item()))
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
    loss_train = compute_loss(mod=model, x=x_train, y=y_train, loss_fn=loss_fn, lambda_val=lambda_penalty)
    loss_val = compute_loss(mod=model, x=x_val, y=y_val, loss_fn=loss_fn, lambda_val=0)

    dlv_dw = grad(loss_val, model.parameters(), create_graph=True)
    dlv_dw = torch.cat([grad.view(-1) for grad in dlv_dw])

    dlt_dw = grad(loss_train, model.parameters(), create_graph=True)
    dlt_dw = torch.cat([x.flatten() for x in dlt_dw])

    # Inner optimization through conjugate gradient method
    with torch.no_grad():
        inverse = conjugate_gradient(dlt_dw, dlv_dw, model, 1e-6, 1000)
        approx = grad(dlt_dw, lambda_penalty, grad_outputs=inverse, retain_graph=True)[0]
        newval = lambda_penalty.item() - l_lr * (0 - approx)
        lambda_penalty.copy_(newval)
        print(lambda_penalty)

