from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()
    # general
    config.seed = 42
    config.steps = 10000
    # hyperparameters
    config.hyperparams = config_dict.ConfigDict({
        "batch_size": 64,
        "embed_dim": 128,
        "heads": 2,
        "blocks": 3,
        "dropout": 0.1,
        "seq_len": 128,
    })
    # io and logging
    config.log_interval = 1000
    config.checkpt_dir = "checkpoints"
    config.save_dir = "results"
    # wandb
    config.use_wandb = False
    config.wandb_entity = "sidharth-baskaran"
    config.wandb_project = "gpt"

    return config
