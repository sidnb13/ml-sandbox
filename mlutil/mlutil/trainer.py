'''
A trainer class for training a model.
Functionality for training, validation, and testing.
'''

from collections import defaultdict

class Trainer:
    def __init__(self, epochs=10) -> None:
        self.epochs = epochs
        self.metrics = defaultdict(list)

    def train(self, model, train_loader, val_loader):
        """Train the model."""
        for epoch in range(self.epochs):
            model.train()
            model.opt.zero_grad()
            train_loss = 0
            for batch in train_loader:
                train_loss += model.train_step(batch)
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            for batch in val_loader:
                val_loss += model.val_step(batch)
            val_loss /= len(val_loader)

            self.metrics['tr_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)

            print(
                f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')

    def plot(self):
        """Plot the training and validation loss."""
        for k, v in self.metrics.items():
            plt.plot(v, label=k)
        plt.legend()
        plt.title('metrics')
        plt.show()
