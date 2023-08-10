import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def data_generating(n):
    xs = np.random.randn(5, n)
    y = 0.5*xs[0] + xs[0]**2 + np.exp(xs[1])/2 + (xs[2]*xs[3])/7 + np.sin(xs[4]) + np.random.randn(n)
    data = pd.DataFrame(xs.T, columns=[f'x{j}' for j in range(1, 5 + 1)])
    data['y'] = y

    return data

def sample_inputs(df, batch_size):
    indices = torch.randperm(len(df))[:batch_size]

    batch_inputs = torch.tensor(df.loc[indices].filter(regex=r'x\d+$').values, dtype=torch.float32)
    batch_outputs = torch.tensor(df.loc[indices].y.values, dtype=torch.float32).reshape(-1, 1)
    return batch_inputs, batch_outputs


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size, sub_nets):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.hidden3 = nn.Linear(hidden2, hidden3)
        self.output = nn.Linear(hidden3, output_size)
        self.relu = nn.ReLU()

        self.subnetwork_ids = self.assign_subnetwork_ids(sub_nets)

    def assign_subnetwork_ids(self, sub_nets):
        subnetwork_ids = {}
        tensors = [param for _, param in self.named_parameters()]

        for name, weight_tensor in self.named_parameters():
            shape = weight_tensor.shape
            subnetwork_ids[name] = np.random.choice(sub_nets, shape)

        return subnetwork_ids

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.output(x)
        return x
N = 100000
train = data_generating(N)
val = data_generating(20000)

# Divide into subsets
K = 10
chunk_size = N // 10
train['idx'] = np.random.randint(K, size=N)
loss_fn = nn.MSELoss()
learning_rate = 1e-3
model = NeuralNetwork(input_size=5, hidden1=32, hidden2=16, hidden3=8, output_size=1, sub_nets=K)  # Instantiate the model

iters = 15
chunk = np.random.choice(range(0, K), iters)
num_params = sum(p.numel() for p in model.parameters())
batch_size = 32
lambda_penalty = torch.tensor(0.05, requires_grad=True)
lambda_penalty = torch.nn.Parameter(lambda_penalty)
lambda_penalty.data = torch.max(lambda_penalty.data, torch.zeros_like(lambda_penalty.data))

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer_phi = optim.SGD([lambda_penalty], lr=learning_rate)

for i in range(iters):
    optimizer.zero_grad()
    optimizer_phi.zero_grad()

    train_sub = train[train['idx'].isin(range(0, chunk[i]+1))].reset_index()
    batch_inputs, batch_outputs = sample_inputs(df = train_sub, batch_size=batch_size)

    outputs = model(batch_inputs)
    loss = loss_fn(outputs, batch_outputs)

    parameters = to,rch.cat([param.flatten() for param in model.parameters()])
    reg_term = 0.5 * lambda_penalty * torch.norm(parameters, p=2) ** 2
    loss += reg_term

    loss.backward()

    # Exclude gradients for weights in excluded subnetworks
    for name, param in model.named_parameters():
        excluded_mask = torch.tensor(model.subnetwork_ids[name]) <= chunk[i]
        param.grad[excluded_mask] = 0.0

    # Update parameters
    optimizer.step()

    # Update the hyperparameter
    k = np.random.choice(range(1, K), 1)
    train_sub = train[train['idx'] == k[0]].reset_index()
    batch_inputs, batch_outputs = sample_inputs(df = train_sub, batch_size=batch_size)
    outputs = model(batch_inputs)
    loss_phi = loss_fn(outputs, batch_outputs)

    parameters = torch.cat([param.flatten() for param in model.parameters()])
    reg_term = 0.5 * lambda_penalty * torch.norm(parameters, p=2) ** 2
    loss_phi += reg_term
    # Step 5e: Compute gradients with respect to lambda_penalty
    loss_phi.backward()
    optimizer_phi.step()
    lambda_penalty = torch.max(lambda_penalty, torch.tensor(0.0))

#    loss.backward()


