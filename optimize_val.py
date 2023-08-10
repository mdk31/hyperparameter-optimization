import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.func import functional_call, vjp, grad
from functools import partial

torch.manual_seed(123)

def data_generating(n):
    xs = np.random.randn(5, n)
    y = 0.5*xs[0] + xs[0]**2 + np.exp(xs[1])/2 + (xs[2]*xs[3])/7 - np.log(abs(xs[3])) + np.sin(xs[4]) + np.random.randn(n)
    return xs.T, y

def compute_loss(params, mod, x, y, loss_fn, lambda_val=0):
    yhat = functional_call(mod, params, x)
    reg_term = torch.norm(torch.cat([param.view(-1) for param in mod.parameters()]), p=2) ** 2
    return loss_fn(yhat, y) + 0.5*lambda_val*reg_term
#
# def compute_loss_lam(lambda_val, params, mod, x, y, loss_fn):
#     yhat = functional_call(mod, params, x)
#     reg_term = torch.norm(torch.cat([param.view(-1) for param in mod.parameters()]), p=2) ** 2
#     return loss_fn(yhat, y) + 0.5*lambda_val*reg_term


n = 10000
val_n = 20000
x_train, y_train = data_generating(n)
x_train = torch.as_tensor(x_train).float()
y_train = torch.as_tensor(y_train).float().view(n, -1)
train_dat = TensorDataset(x_train, y_train)

x_val, y_val = data_generating(val_n)
x_val = torch.as_tensor(x_val).float()
y_val = torch.as_tensor(y_val).float().view(val_n, -1)


batch_size = 32
train_loader = DataLoader(dataset=train_dat,
                          shuffle = True,
                          batch_size=batch_size)

lr = 0.01
beta = 0.001
torch.manual_seed(123)
model = nn.Sequential(nn.Linear(5, 16),
                      nn.ReLU(),
                      nn.Linear(16, 8),
                      nn.ReLU(),
                      nn.Linear(8, 1))
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
lambda_penalty = torch.nn.Parameter(torch.tensor(5.0, requires_grad=True))

compute_loss_train = partial(compute_loss, mod=model, x=x_train, y=y_train, loss_fn=loss_fn, lambda_val=lambda_penalty)
compute_loss_val = partial(compute_loss, mod=model, x=x_val, y=y_val, loss_fn=loss_fn, lambda_val=0)
compute_loss_lam_ex = partial(compute_loss_lam, params = dict(model.named_parameters()), mod=model, x=x_train, y=y_train, loss_fn=loss_fn)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch
        y_batch = y_batch

        model.train()
        optimizer.zero_grad()

        y_pred = model(x_batch)
        # Compute penalized loss
        regularization_term = 1 / 2 * lambda_penalty * torch.norm(torch.cat([param.view(-1) for param in model.parameters()]),
                                                         p=2) ** 2
        loss = loss_fn(y_pred, y_batch) + regularization_term
        loss.backward()
        optimizer.step()


    # Update lambda
    ## Calculate new loss
    optimizer.zero_grad()
    regularization_term = 1 / 2 * lambda_penalty * torch.norm(torch.cat([param.view(-1) for param in model.parameters()]), p=2) ** 2
    y_train_pred = model(x_train)
    y_val_pred = model(x_val)
    loss_train = loss_fn(y_train_pred, y_train) + regularization_term
    loss_val = loss_fn(y_val_pred, y_val) + regularization_term

    # dlv_dw = grad(loss_val, model.parameters(), create_graph=True)
    # dlv_dw = torch.cat([grad.view(-1) for grad in dlv_dw])
    # dlt_dw = grad(loss_train, model.parameters(), create_graph=True)
    # dlt_dw = torch.cat([grad.view(-1) for grad in dlt_dw])

    dlv_dw = grad(compute_loss_val)(dict(model.named_parameters()))
    _, vjpfunc = vjp(grad(compute_loss_train), dict(model.named_parameters()))

    with torch.no_grad():
        v = {key: param.detach().clone() for key, param in dlv_dw.items()}
        p = {key: param.detach().clone() for key, param in dlv_dw.items()}
        # v = dlv_dw.detach().clone()
        # p = dlv_dw.detach().clone()
        for i in range(75):
            tmp_v = vjpfunc(v)[0]
            # tmp_v = grad(dlt_dw, model.parameters(), grad_outputs=v, retain_graph=True)
            # tmp_v = torch.cat([grad.view(-1) for grad in tmp_v])
            v = {key: tmp_v[key] - v[key] for key in tmp_v}
            p = {key: p[key] + v[key] for key in tmp_v}

        p = torch.cat([x.view(-1) for x in p.values()])


        # print(p['0.weight'])
        # dlv_dlambda = grad(compute_loss_lam_ex)(lambda_penalty)
    dlt_dw = autograd.grad(loss_train, model.parameters(), create_graph=True)
    dlt_dw = torch.cat([x.flatten() for x in dlt_dw])
    dlt_dlambda = autograd.grad(loss_train, lambda_penalty, retain_graph=True)[0]
    v3 = autograd.grad(dlt_dw, lambda_penalty, grad_outputs=p, retain_graph=True)
    print(v3)
        # dlv_dlambda = grad(loss_val, lambda_penalty)[0]
        # newval = lambda_penalty.item() + dlv_dlambda - v3
        # print(newval)
        # lambda_penalty.copy_(newval)

        #lambda_penalty = lambda_penalty - 0.01*(dlv_dlambda - v3)

    #
    # ## Inverse HVP
    # # Initialize
    # p = []
    #
    #
    #
    #     dlt_dw = grad(loss, model.parameters(), create_graph=True)
    #     dlt_dw = torch.cat([grad.view(-1) for grad in dlt_dw])
    #
    #     loss_val = loss_fn(y_val_pred, y_val) + reg_val
    #
    #     dlv_dlamda = grad(loss_val, lambda_penalty, create_graph=True)[0]
    #     # Hypergradient
    #     ## Inverse HVP
    #     dlv_dw = grad(loss_val, model.parameters(), create_graph=True)
    #     dlv_dw = torch.cat([grad.view(-1) for grad in dlv_dw])
    #     p = dlv_dw.detach().clone()
    #     with torch.no_grad():
    #         for i in range(75):
    #             update = grad(dlt_dw, model.parameters(), grad_outputs=p, create_graph=True)
    #             update = torch.cat([grad.view(-1) for grad in update])
    #             dlv_dw -= beta * update
    #             p += dlv_dw
    #         v3 = grad(dlt_dw, lambda_penalty, grad_outputs=p, retain_graph=True)[0]
    #         lambda_penalty -= beta * (dlv_dlamda - v3)
    #
    #     # Update weight parameters
    #
    #     print(lambda_penalty)
    #
    #     if lambda_penalty < 0:
    #         lambda_penalty = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))


