from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    # general
    config.seed = 38
    config.steps = 10000
    config.special_tokens = ["<unk>", "<sos>", "<eos>", "<pad>", "<mask>"]
    # hyperparameters
    config.hyperparams = config_dict.ConfigDict(
        {
            "batch_size": 32,
            "embed_dim": 128,
            "heads": 4,
            "blocks": 6,
            "dropout": 0.1,
            "block_size": 64,
        }
    )
    config.warmup_steps = 4000
    config.betas = (0.9, 0.98)
    config.eps = 1e-9
    config.load_model = False
    # io and logging
    config.log_interval = 1000
    config.checkpt_dir = "checkpoints"
    config.save_dir = "results"
    # wandb
    config.use_wandb = False
    config.wandb_entity = "sidharth-baskaran"
    config.wandb_project = "gpt"

    return config
