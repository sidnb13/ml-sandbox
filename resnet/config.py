# Config file for ResNet implementation

import pathlib

num_classes = 10
batch_size = 128
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
epochs = 50
log_interval = 1
metric_res = 32
checkpt_interval = 8

data_dir = "./data/cifar10"
checkpt_dir = pathlib.Path(__file__).parent.absolute() / "checkpoints"
save_dir = pathlib.Path(__file__).parent.absolute() / "results"