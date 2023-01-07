import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import kaggle
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True


class ProcessDataset:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_kaggle_dataset():
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "sagyamthapa/handwritten-math-symbols", path=".data/", unzip=True
        )

    @staticmethod
    def perform_preprocessing():
        # resize all images to 45x45 size optimize
        dataset_path = "./dataset"
        for image_file in glob.glob(os.path.join(dataset_path, "**/*")):
            if image_file.endswith(".jpg") or image_file.endswith(".png"):
                img = Image.open(image_file)
                img = img.resize((45, 45)).convert("L")
                img.save(image_file)

        label_pairs = []
        labels_map = {}

        p = Path(dataset_path)
        class_dirs = [f.__str__() for f in p.iterdir() if f.is_dir()]

        for i, class_name_dir in enumerate(class_dirs):
            class_name = os.path.basename(class_name_dir)
            for image in glob.glob(os.path.join(class_name_dir, "*")):
                label_pairs.append((image, i))

            labels_map[i] = class_name

        with open("./dataset/latex_handwritten_labels.csv", "w") as f:
            for pair in label_pairs:
                f.write(f"{pair[0]}, {pair[1]}\n")

        return labels_map

    @staticmethod
    def get_train_val_test_splits(dataset, splits=(0.7, 0.2, 0.1)):
        train_size = int(splits[0] * len(dataset))
        val_size = int(splits[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size

        print(sum([train_size, val_size, test_size]))

        train_set, test_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        return train_set, val_set, test_set


class LatexHandwrittenDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_data():
    label_map = ProcessDataset.perform_preprocessing()

    symbol_dataset = LatexHandwrittenDataset(
        "./dataset/latex_handwritten_labels.csv",
        "",
        transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
        target_transform=transforms.Lambda(
            lambda y: torch.zeros(len(label_map.keys()), dtype=torch.float).scatter_(
                dim=0, index=torch.tensor(y), value=1
            )
        ),
    )

    train_set, val_set, test_set = ProcessDataset.get_train_val_test_splits(
        symbol_dataset
    )

    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.basic_neural_stack = nn.Sequential(
            nn.Linear(45 * 45, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 19),  # 19 classes
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.basic_neural_stack(x)
        return logits


class TrainModel:
    def __init__(
        self, train_loader, val_loader, test_loader, optimizer, loss_fn, model=NeuralNet
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loss = []
        self.val_loss, self.val_accuracy = [], []
        self.test_loss, self.test_accuracy = [], []

    def train_loop(self):
        size = len(self.train_loader.dataset)

        # dont loop through dataloader.dataset will screw up batches
        for batch, (X, y) in enumerate(self.train_loader):
            if DEBUG:
                print(f"TRAIN BATCH: X: {X.size()}, y: {y.size()}")

            pred = self.model(X)

            if DEBUG:
                print(f"PRED: {pred.shape}, Y: {y.shape}")

            loss = self.loss_fn(pred, y)

            # Perform backprop
            self.optimizer.zero_grad()  # prevent cumulative gradients
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)

                # perform validation set evaluation
                val_loss_curr, val_acc_curr = self.validation_loop()
                # perform test set evaluation
                test_loss_curr, test_acc_curr = self.test_loop()

                print(f"[train loss]: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(
                    f"[val acc]: {(val_acc_curr):>0.1f}%, [val loss (avg)]: {val_loss_curr:>8f}"
                )
                print(
                    f"[test acc]: {(test_acc_curr):>0.1f}%, [test loss (avg)]: {test_loss_curr:>8f} \n"
                )

                self.train_loss.append(loss)
                self.val_loss.append(val_loss_curr)
                self.val_accuracy.append(val_acc_curr)
                self.test_loss.append(test_loss_curr)
                self.test_accuracy.append(test_acc_curr)

    def validation_loop(self):
        correct_preds = total = 0
        running_loss = 0.0

        self.model.eval()

        with torch.no_grad():
            for X, y in self.val_loader:
                if DEBUG:
                    print(f"VAL BATCH: X: {X.size()}, y: {y.size()}")

                output = self.model(X)
                total += y.size(0)

                # convert argmax output to onehot encoding
                pred_argmax = output.argmax(1)
                pred_onehot = torch.zeros_like(y).scatter_(
                    1, pred_argmax.unsqueeze(1), 1
                )

                correct_preds += (pred_onehot == y).type(torch.float).sum().item()
                running_loss += self.loss_fn(output, y)

        # compute avg loss
        running_loss /= len(self.val_loader)
        acc = 100 * correct_preds / total

        self.model.train()

        return running_loss, acc

    def test_loop(self):
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss = total = correct = 0

        self.model.eval()

        with torch.no_grad():
            for X, y in self.test_loader:
                if DEBUG:
                    print(f"TEST BATCH: X: {X.size()}, y: {y.size()}")

                pred = self.model(X)
                total += y.size(0)
                test_loss += self.loss_fn(pred, y).item()

                # convert argmax output to onehot encoding
                pred_argmax = pred.argmax(1)
                pred_onehot = torch.zeros_like(y).scatter_(
                    1, pred_argmax.unsqueeze(1), 1
                )

                correct += (pred_onehot == y).type(torch.float).sum().item()

        # compute average test loss
        test_loss /= num_batches
        # accuracy
        correct /= size

        self.model.train()

        return test_loss, 100 * correct


def peruse_dataset(test_dataloader):
    train_features, train_labels = next(iter(test_dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    img = train_features[0].squeeze()
    label = train_labels[0]

    plt.imshow(img, cmap="gray")
    plt.title(f"class name: {label_map[torch.argmax(label).item()]}")
    plt.show()
    print(f"Label: {label}, Class name: {label_map[torch.argmax(label).item()]}")


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = get_data()

    hyperparams = {"batch_size": 64, "lr": 1e-3, "epochs": 10}

    model = NeuralNet().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams["lr"])
    loss_fn = nn.CrossEntropyLoss()

    model_trainer = TrainModel(
        train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, model
    )

    for epoch in range(hyperparams["epochs"]):
        print(f"EPOCH {epoch + 1} -------------------")
        model_trainer.train_loop()

    train_lossh = model_trainer.train_loss
    val_lossh = model_trainer.val_loss
    test_lossh = model_trainer.test_loss

    epoch_range = np.arange(1, len(train_lossh) + 1, 1)

    plt.plot(epoch_range, train_lossh, label="train loss")
    plt.plot(epoch_range, val_lossh, label="val loss")
    plt.plot(epoch_range, test_lossh, label="test loss")

    plt.legend()
    plt.show()
