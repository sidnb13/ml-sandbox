import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from typing import Optional

def random_data(ds_size: int = 512, n: int = 32, train_split: float=0.9, val_split: float=0.1) -> None:
    X = torch.rand(ds_size, n)
    noise = torch.randn(ds_size, 1) * 0.01
    w, b = torch.ones((n , 1)) * 0.01, 0.05
    y = torch.matmul(X, w) + b + noise
        
    train_slice = int(ds_size * train_split)
    val_slice = int(ds_size * (val_split + train_split))
        
    return (X[:train_slice], y[:train_slice]),\
            (X[train_slice:val_slice], y[train_slice:val_slice]),\
            (X[val_slice:], y[val_slice:])

class CustomData(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        super().__init__()
                
        self.X = X
        self.y = y
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> None:
        return self.X[idx], self.y[idx]

class LinearRegressionDecay(nn.Module):
    def __init__(self, num_inputs: int, lambd: float, sigma: float=0.01) -> None:
        super().__init__()
        # Sample weights from a Gaussian distribution
        self.w = torch.normal(0, sigma, size=(num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.lambd = lambd
        
    def loss(self, y_hat, y):
        return torch.mean((y_hat - y) ** 2) / 2 + self.lambd * torch.sum(self.w ** 2) / 2
    
    def forward(self, X):
        return torch.matmul(X, self.w) + self.b
    
if __name__ == '__main__':
    # define dataset
    num_inputs = 32
    (X_train, y_train), (X_val, y_val), (X_test, Y_test) = random_data(ds_size=512, n=num_inputs, train_split=0.8, val_split=0.1)
    
    train_data = CustomData(X_train, y_train)
    test_data = CustomData(X_test, Y_test)
    val_data = CustomData(X_val, y_val)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # training
    model = LinearRegressionDecay(num_inputs=num_inputs, lambd=0.01)
    optimizer = torch.optim.SGD([model.w, model.b], lr=0.01)
    epochs = 10
    
    train_loss, val_loss, test_loss = [], [], []
    
    def eval_loop(loader: DataLoader, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]=None, train: bool=False):
        total_loss = 0
        for X, y in loader:
            if train and optimizer is not None:
                optimizer.zero_grad()
                y_hat = model(X)
                loss = model.loss(y_hat, y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    y_hat = model(X)
                    total_loss += model.loss(y_hat, y).item()
        return total_loss / len(loader)
    
    for epoch in range(epochs):
        train_loss.append(eval_loop(train_loader, model, optimizer, train=True))
        test_loss.append(eval_loop(test_loader, model))
        val_loss.append(eval_loop(val_loader, model))
        
        print('Epoch %d | train loss: %.4f, test loss: %.4f, val loss: %.4f, L2_w: %.4f' % 
              (epoch, train_loss[-1], test_loss[-1], val_loss[-1], torch.sum(model.w ** 2) / 2))
    
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.plot(val_loss, label='val')
    
    plt.legend()
    plt.yscale('log')
    plt.show()