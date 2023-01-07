import pathlib
from datetime import datetime

# hyperparameters
num_classes = 10
batch_size = 100
lr = 1e-2
steps = 10_000

# vae params
input_dim = 784
latent_dim = 5
hidden_dim = 512

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
log_interval = 1000
test_interval = 1000
checkpt_interval = 8
gen_samples = 10
manifold_dim = 9

data_dir = "./data/mnist"
result_dir = "./data/vae/results"
checkpt_dir = pathlib.Path(result_dir) / "checkpoints"
save_dir = pathlib.Path(result_dir) / "results"
job_id = datetime.strftime(datetime.now(), "%m-%d-%y-%H-%M-%S")
