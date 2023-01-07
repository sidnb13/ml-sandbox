import pathlib
from datetime import datetime

# hyperparameters
num_classes = 10
batch_size = 100
lr = 1e-3
steps = 100_000

# vae params
input_dim = 784
latent_dim = 2
hidden_dim = 500

# general config
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
log_interval = 100
test_interval = 1000
checkpt_interval = 8

data_dir = "./data/mnist"
result_dir = "./data/vae/results"
checkpt_dir = pathlib.Path(result_dir) / "checkpoints"
save_dir = pathlib.Path(result_dir) / "results"
job_id = datetime.strftime(datetime.now(), "%m-%d-%y-%H-%M-%S")
