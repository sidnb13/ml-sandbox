import pathlib

batch_size = 128
epochs = 1
log_interval = 1
metric_res = 8
checkpt_interval = 8

data_dir = "./data/cifar10"
checkpt_dir = pathlib.Path(__file__).parent.absolute() / "checkpoints"
save_dir = pathlib.Path(__file__).parent.absolute() / "results"