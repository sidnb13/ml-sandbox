import pathlib
from datetime import datetime

# hyperparameters
num_classes = 10
batch_size = 28
lr = 1e-2
init_momentum = 0.5
steps = 100

# model
k_steps = 1
latent_dim = 100
image_size = (28, 28)
num_blocks = 3
hidden_dims = [128, 256, 512, 784]

# io and logging
log_interval = 10
data_dir = "./data/mnist"
result_dir = "./data/gan/results"
checkpt_dir = pathlib.Path(result_dir) / "checkpoints"
save_dir = pathlib.Path(result_dir) / "results"
job_id = datetime.strftime(datetime.now(), "gan-%m-%d-%y-%H-%M-%S")

# wandb
use_wandb = True
wandb_entity = "sidharth-baskaran"
wandb_project = "gan"
