import matplotlib.pyplot as plt
import numpy as np
import torch


class SyntheticData:
    def __init__(
        self, w, b, noise=0.01, train_examples=1000, val_examples=100, batch=32
    ) -> None:
        self.w = w
        self.b = b
        self.batch = batch
        self.epsilon = torch.randn(train_examples + val_examples) * noise
        self.X = torch.rand(train_examples + val_examples, len(w))
        self.y = self.X @ self.w + self.b + self.epsilon
        self.val_X = torch.rand(val_examples, len(w))
        self.val_y = self.val_X @ self.w + self.b + self.epsilon[:val_examples]

    def get_dataloader(self, train=True, batch_size=32):
        # Shuffle if we're using the training set
        if train:
            indices = torch.randperm(len(self.X))
        else:
            indices = torch.arange(len(self.val_X))
        for i in range(0, len(indices), self.batch):
            batch_indices = torch.tensor(indices[i : i + self.batch])
            if train:
                yield self.X[batch_indices], self.y[batch_indices]
            else:
                yield self.val_X[batch_indices], self.val_y[batch_indices]


class LinearRegression:
    def __init__(self, input_dim, lr=0.01, sigma=0.01) -> None:
        self.w = torch.normal(0, sigma, size=(input_dim, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.lr = lr

    def forward(self, X):
        return torch.mm(X, self.w) + self.b

    def loss(self, y_hat, y):
        return torch.mean((y_hat - y) ** 2 / 2)


class SGDOpt:
    def __init__(self, params, lr) -> None:
        self.params = params
        self.lr = lr

    def step(self):
        # need no grad in order to update the parameters in place
        # non-inplace update would make the parameters non-leaf nodes
        with torch.no_grad():
            for p in self.params:
                p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


def train(model, data, epochs=10, lr=0.01):
    opt = SGDOpt([model.w, model.b], lr)

    def train_epoch():
        i = 0
        total_loss = 0

        for X, y in data.get_dataloader():
            y_hat = model.forward(X)
            loss = model.loss(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            i += 1
            total_loss += loss.item()

        return total_loss / i

    def val_epoch():
        i = 0
        total_loss = 0

        with torch.no_grad():
            for X, y in data.get_dataloader(train=False):
                y_hat = model.forward(X)
                loss = model.loss(y_hat, y)
                i += 1
                total_loss += loss.item()

        return total_loss / i

    train_loss, val_loss = [], []

    for epoch in range(epochs):
        train = train_epoch()
        val = val_epoch()
        print(f"Epoch {epoch} | Train Loss: {train_loss} | Validation Loss: {val_loss}")
        train_loss.append(train)
        val_loss.append(val)

    return train_loss, val_loss


if __name__ == "__main__":
    w = torch.tensor([1, 2, 3], dtype=torch.float32)
    b = torch.tensor([1], dtype=torch.float32)
    data = SyntheticData(w, b)
    model = LinearRegression(len(w))
    train_loss, val_loss = train(model, data)

    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.show()
