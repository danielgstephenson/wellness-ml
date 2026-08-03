import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import os
import torch
from torch.utils.data import TensorDataset, random_split
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
print(os.getcwd())
controls = pd.read_csv('../data/clean/controls_scaled.csv').to_numpy()
outcomes = pd.read_csv('../data/clean/outcomes.csv')['spend_0816_0717'].to_numpy()
treatment = pd.read_csv('../data/clean/treatment.csv').to_numpy()
X = torch.tensor(controls).float()
y = torch.tensor(outcomes).float()

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = F.relu
        self.W = 10
        self.H = 3
        self.project_layer = nn.Linear(X.shape[1], self.W)
        self.extra_hidden_layers = nn.ModuleList()
        for _ in range(self.H-1):
            self.extra_hidden_layers.append(nn.Linear(self.W,self.W))
        self.final_layer = nn.Linear(self.W, 1)
    def forward(self, x: Tensor) -> Tensor:
        x = self.project_layer(x)
        for i in range(self.H-1):
            h = self.extra_hidden_layers[i]
            x = x + self.activation(h(x))
        x = self.final_layer(x)
        return x.squeeze(1)
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)


dataset = TensorDataset(X,y)

# Start Loop Here

train_set, test_set = random_split(dataset, [0.5, 0.5])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
train_X, train_y = next(iter(train_loader))

model = Model()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
alpha = 0.01

folds = random_split(train_set, [0.1 for i in range(10)])

# for step in range(100000):
#     output = model(train_X)
#     mse = torch.mean((train_y-output)**2)
#     L1 = torch.tensor(0., requires_grad=True)
#     for param in model.parameters():
#         L1 = L1 + torch.norm(param, 1)
#     loss = mse + alpha*L1
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     if step % 100 == 0:
#         print('loss',f'{loss.item():.3f}',step)



