from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    # general
    config.seed = 38
    # data
    config.data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    config.train_split = 0.9
    # hyperparameters
    config.hyperparams = config_dict.ConfigDict(
        {
            "batch_size": 32,
            "embed_dim": 128,
            "heads": 4,
            "blocks": 4,
            "dropout": 0.1,
            "block_size": 64,
        }
    )
    config.warmup_steps = 2000
    config.steps = 10000
    config.betas = (0.9, 0.999)
    config.eps = 1e-9
    config.load_model = False
    # io and logging
    config.log_interval = 10
    config.checkpt_dir = "checkpoints"
    config.checkpt_name = "gpt-mini-test.pt"
    config.save_dir = "results"
    # wandb
    config.use_wandb = False
    config.wandb_entity = "sidharth-baskaran"
    config.wandb_project = "gpt"

    return config
