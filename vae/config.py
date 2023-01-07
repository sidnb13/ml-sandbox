import pathlib
from datetime import datetime

# hyperparameters
num_classes: int = 10
batch_size: int = 64
lr: float = 1e-3
epochs: int = 10

# vae params
latent_dim = 2
hidden_dim = 512

# general config
classes: int = (
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
log_interval: int = 1
metric_res: int = 32
checkpt_interval: int = 8
data_dir: str = "./data/cifar10"
checkpt_dir: str = pathlib.Path(__file__).parent.absolute() / "checkpoints"
save_dir: str = pathlib.Path(__file__).parent.absolute() / "results"
job_id = datetime.strftime(datetime.now(), "%m-%d-%y-%H-%M-%S")
