import pathlib
from datetime import datetime

# hyperparameters
num_classes = 10
batch_size = 28
lr = 1e-3
init_momentum = 0.5
steps = 10000

# model
k_steps = 1
latent_dim = 100
image_size = (28, 28)
num_blocks = 3
hidden_dims = [128, 256, 512]

# io and logging
log_interval = 1000
data_dir = "/Users/sidbaskaran/code/ml-sandbox/data/mnist"
result_dir = "/Users/sidbaskaran/code/ml-sandbox/data/gan"
checkpt_dir = pathlib.Path(result_dir) / "checkpoints"
save_dir = pathlib.Path(result_dir) / "results"
job_id = datetime.strftime(datetime.now(), "gan-%m-%d-%y-%H-%M-%S")

# wandb
use_wandb = True
wandb_entity = "sidharth-baskaran"
wandb_project = "gan"
