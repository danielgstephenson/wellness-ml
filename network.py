import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
print(os.getcwd())
controls = pd.read_csv('data/clean/controls_scaled.csv').to_numpy()
outcomes = pd.read_csv('data/clean/outcomes.csv')['spend_0816_0717'].to_numpy()
treatment = pd.read_csv('data/clean/treatment.csv').to_numpy()
controls_tensor = torch.tensor(controls).float()
outcomes_tensor = torch.tensor(outcomes).float()
full_dataset = TensorDataset(controls_tensor, outcomes_tensor)
print(outcomes_tensor.shape)

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = F.relu
        self.W = 10
        self.H = 3
        self.project_layer = nn.Linear(controls_tensor.shape[1], self.W)
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

# Start Loop Here

def split[T](lst: list[T], n: int)->list[list[T]]:
    shuffled_lst = lst.copy()
    random.shuffle(shuffled_lst)
    return [shuffled_lst[i::n] for i in range(n)]

dataset_indices = [i for i in range(len(full_dataset))]
train_indices, test_indices = split(dataset_indices, 2)
train_set = TensorDataset(controls_tensor[train_indices], outcomes_tensor[train_indices])
test_set = TensorDataset(controls_tensor[test_indices], outcomes_tensor[test_indices])

fold_count = 2
test_folds = split(train_indices, fold_count)
train_folds = [
    list(set(dataset_indices)-set(fold_indices)) 
    for fold_indices in test_folds
]

def train(dataloader: DataLoader[tuple[Tensor,...]], alpha: float)->Model:
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for epoch in range(50):
        for batch in dataloader:
            data: tuple[Tensor,...] = batch
            X, y = data
            output = model(X)
            mse = torch.mean((y-output)**2)
            L1 = torch.tensor(0.0, requires_grad=True)
            for param in model.parameters():
                L1 = L1 + torch.norm(param, 1)
            loss = mse + alpha*L1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # SSR = torch.sum((y-output)**2)
            # mean_y = torch.mean(y)
            # SST = sum((y-mean_y)**2)
            # R2 = 1 - SSR/SST
        R2 = test(model, dataloader.dataset[:])
        print(f'epoch: {epoch}, R2: {R2:.4f}')
    return model

def test(model: Model, data: tuple[Tensor,...])->float:
    X, y = data
    output = model(X)
    SSR = torch.sum((y-output)**2)
    mean_y = torch.mean(y)
    SST = sum((y-mean_y)**2)
    R2 = 1 - SSR/SST
    return R2.item()

grid = np.array([i for i in range(2)])
alpha_grid = (1*np.exp(0.2*grid))
R2_grid = 0*alpha_grid

file = open('alpha_r2.csv',"a",buffering=1,encoding="utf-8")
file.writelines(f'alpha,R2\n')

for i in grid:
    alpha = float(alpha_grid[i])
    print(f'Check alpha {alpha:0.4f}')
    R2s = []
    for k in range(fold_count):
        train_fold_dataset = TensorDataset(controls_tensor[train_folds[k]], outcomes_tensor[train_folds[k]])
        train_dataloader = DataLoader(train_fold_dataset, batch_size=32, shuffle=True)   
        model = train(train_dataloader,alpha)
        test_fold_dataset = TensorDataset(controls_tensor[test_folds[k]], outcomes_tensor[test_folds[k]])
        test_fold_data = test_fold_dataset.tensors
        R2 = test(model, test_fold_data)
        R2s.append(R2)
        print(f'R2: {R2:.4f}, Fold {k}, alpha {alpha:.4f}')
    mean_R2 = np.mean(R2s)
    print(f'R2: {mean_R2:.4f}, alpha {alpha:.4f}')
    file.writelines(f'{alpha:0.4f},{mean_R2:0.4f}\n')





