# Config file for ResNet implementation

import pathlib

num_classes = 10
batch_size = 64
lr = 1e-2
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
epochs = 10
log_interval = 1
eval_interval = 1
checkpt_interval = 8

data_dir = "./data/cifar10"
checkpt_dir = pathlib.Path(__file__).parent.absolute() / "checkpoints"