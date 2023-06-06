"""
Small decoder-only transfomer for autoregressive text generation.
"""

import datetime
import pathlib
import time
from itertools import cycle

import torch
import wandb
from absl import app, flags, logging
from ml_collections import config_dict, config_flags
from tiktoken.core import Encoding
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import SimpleTextDataset, Trainer, setup_device

FLAGS = flags.FLAGS

# define configuration file
config_flags.DEFINE_config_file(
    "config",
    None,
    "Config file for training.",
    lock_config=True,
)

# define command line flags
flags.DEFINE_enum(
    "run_mode",
    "train",
    ["train", "generate", "debug"],
    "Mode to run GPT model in.",
)
flags.DEFINE_string(
    "prompt",
    None,
    "Prompt to use for autoregressive generation.",
)
flags.DEFINE_integer("gen_len", 128, "Sequence length for generation in tokens.")


class DecoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        block_size: int,
        attn_dropout: float,
        layer_dropout: float,
        return_attn_weights: bool = False,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        # positionwise FFN
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),  # as in original GPT
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(layer_dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.return_attn_weights = return_attn_weights

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # self attention and residual, -1 denotes padding
        _, T, _ = x.shape
        causal_mask = torch.tril(torch.ones(T, T)).to(x.device)
        attn_out, attn_weights = self.multihead_attention(
            x,
            x,
            x,
            attn_mask=causal_mask.to(torch.bool),
            key_padding_mask=attn_mask.to(torch.bool),
        )
        x = self.ln1(x + attn_out)
        # feed forward and residual
        x = self.ln2(x + self.feed_forward(x))

        if self.return_attn_weights:
            return x, attn_weights
        else:
            return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        blocks: int,
        num_heads: int,
        embed_dim: int,
        block_size: int,
        attn_dropout: float,
        layer_dropout: float,
    ) -> None:
        """Transformer based on decoder blocks.

        Args:
            vocab_size (int): |V| where V is vocab set.
            blocks (int): number of decoder blocks to stack.
            num_heads (int): number of attention heads.
            embed_dim (int): the embedding dimension.
            block_size (int): the number of tokens per input.
            attn_dropout (float): the dropout rate for attention layers.
            layer_dropout (float): the dropout rate for the residual layers.
        """
        super().__init__()
        self.stack_count = blocks
        # add 2 for <sos> token
        self.block_size = block_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        # word embedding layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding(self.pos_embedding.weight)
        # decoder stack
        self.decoder_stack = nn.ModuleList(
            [
                DecoderBlock(
                    self.num_heads,
                    self.embed_dim,
                    self.block_size,
                    attn_dropout=attn_dropout,
                    layer_dropout=layer_dropout,
                    return_attn_weights=(i == blocks - 1),
                )
                for i in range(blocks)
            ]
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.ln = nn.LayerNorm(embed_dim)
        # weight tie between embedding and linear weights
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.linear.weight = self.embedding.weight

    def positional_encoding(self, embed_weights: torch.Tensor):
        # create positional encodings
        pos = torch.arange(0, self.vocab_size, dtype=torch.float)
        pos_encoding = torch.stack(
            [
                pos / pow(10000, (2 * dim // 2) / self.embed_dim)
                for dim in range(self.embed_dim)
            ],
        ).transpose_(0, 1)
        # assign positional encodings as non-trainable weights
        embed_weights.requires_grad = False
        embed_weights[:, 0::2] = torch.sin(pos_encoding[:, 0::2])
        embed_weights[:, 1::2] = torch.cos(pos_encoding[:, 1::2])

    def forward(
        self, idx: torch.Tensor, bypass_embedding: bool = False
    ) -> torch.Tensor:
        # create mask for padding
        B, T = idx.size()
        assert T == self.block_size, "input sequence length mismatch"
        attn_mask = idx == -1
        # create embeddings
        if bypass_embedding:
            input_embed = torch.zeros((B, T, self.embed_dim), device=idx.device)
        else:
            tok_embed = self.embedding(idx) * self.embed_dim**0.5
            pos_embed = self.pos_embedding(idx)
            input_embed = tok_embed + pos_embed

        # feed to transformer
        x = self.dropout(input_embed)
        out = x
        for i in range(len(self.decoder_stack)):
            results = self.decoder_stack[i](out, attn_mask)
            if i == len(self.decoder_stack) - 1:
                out, attn_weights = results
            else:
                out = results
        x = self.ln(x)
        x = self.linear(x)
        return x, attn_weights

    def generate(
        self,
        prompt: torch.Tensor,
        steps: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ):
        """Autoregressively generate tokens from a prompt.
        Uses combination of top-k and top-p nucleus sampling.

        Args:
            prompt (torch.Tensor): input_ids of shape (B, T).
            steps (int): number of tokens to generate.
            temperature (float, optional): sampling temp. Defaults to 1.0.
            top_k (int, optional): top-k k value. Defaults to None.
            top_p (float, optional): top-p nucleus probability. Defaults to 0.9.
        """
        _, T = prompt.size()
        assert T < steps, "prompt length must be less than steps"
        padded_prompt = F.pad(
            prompt,
            (0, ((steps + T) // self.block_size + 1) * self.block_size - T + steps),
            value=-1,
            mode="constant",
        )

        self.eval()

        # define context window size
        window_start, window_end = 0, self.block_size

        # autoregressively generate tokens
        for i in range(T, T + steps):
            x = padded_prompt[:, window_start:window_end]

            mask = x > -1
            if torch.all(mask):
                window_start += 1
                window_end += 1

            last_token = len(x[mask]) - 1

            logits, _ = self.forward(
                x,
                bypass_embedding=False,
            )

            # only use last token logits, ignoring pad
            logits = logits[:, last_token, :] / temperature

            # top-k sampling
            if top_k > 0:
                top_k_indices = torch.topk(logits, top_k, largest=False, dim=-1).indices
                logits[top_k_indices] = -float("inf")

            # nucleus sampling
            if 1 > top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_idx = cum_probs > top_p
                # correct removal indices to keep first token
                remove_idx[:, 0] = 0
                remove_idx[:, 1:] = remove_idx[:, :-1].clone()
                removal_indices = sorted_indices[remove_idx]
                logits[removal_indices] = -float("inf")

            # sample from filtered distribution
            dist = F.softmax(logits, dim=-1)
            next_token_idx = torch.multinomial(dist, 1).squeeze()
            padded_prompt[:, i] = next_token_idx

        return padded_prompt.squeeze()


def init_test(model: Transformer, device: torch.device):
    """Make sure the model is initialized correctly."""
    # get some basic model stats
    num_params = sum(p.numel() for p in model.parameters())
    model.eval()
    sample = torch.zeros(1, 4, dtype=torch.long, device=device)

    # generate random text
    logging.info(f"Sample: {model.generate(sample, 4)}")

    # check loss at init
    ip = torch.zeros(1, model.block_size, device=device, dtype=torch.long)
    loss = Trainer.loss_fn(
        model, {"input": ip, "target": ip}, device, bypass_embedding=True
    )
    random_init_loss = -torch.log(torch.tensor(1 / model.vocab_size))
    logging.info(f"Number of parameters: {num_params}")

    # calculate size of the model
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    logging.info("model size: {:.3f}MB".format(size_all_mb))
    logging.info(f"loss: {loss} | random_init: {random_init_loss}")

    assert abs(loss.item() - random_init_loss) < 1e-5, "softmax loss check failed"


def train(
    model: Transformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
):
    """Train the model.

    Args:
        model (Transformer): model to train
        train_loader (DataLoader): the training data loader
        device (torch.device): device to train on
        config (dict): source of truth
    """
    # create trainer
    trainer = Trainer(model, config, device)
    # train model
    train_iter = cycle(iter(train_loader))
    val_iter = cycle(iter(val_loader))
    # model load logic
    if config.checkpt_name is None:
        config.checkpt_name = f"checkpt-{datetime.datetime.now().timestamp()}.pt"
    # gpt hack
    model_path = pathlib.Path(config.checkpt_dir) / config.checkpt_name
    if config.checkpt_dir is not None and config.load_model:
        if not model_path.exists():
            raise ValueError(f"Model path {model_path} does not exist.")
        logging.info(f"Loading model from {model_path}")
        trainer.load_checkpoint(model_path)
    start_step = max(trainer.opt._step, 0)
    # train loop
    for step in range(start_step, config.steps):
        batch = next(train_iter)
        start = time.time()
        loss = trainer.train_step(batch)
        end = time.time()
        # eval
        if step % config.log_interval == 0:
            val_batch = next(val_iter)
            with torch.no_grad():
                val_loss = trainer.eval_step(val_batch)

            log_params = {
                "time": end - start,
                "train_loss": loss,
                "val_loss": val_loss,
                "lr": trainer.opt._lr(),
                "step": step,
            }

            logging.info(
                "step: {step:07} |\ttrain loss: {train_loss:3f} |\tval loss: {val_loss:3f} |\tlr: {lr:6f} |\ttime: {time:2f}".format_map(
                    log_params
                )
            )
            if config.use_wandb:
                wandb.log(log_params)
                wandb.save(str(model_path))
            trainer.save_checkpoint(model_path)


def sample(
    model: Transformer,
    prompt: str,
    encoding: Encoding,
    config: config_dict,
    device: torch.device,
):
    """Decode a prompt using the model.

    Args:
        model (Transformer): _description_
        prompt (str): string prompt to start generation
        encoding (Encoding): BPE tokenizer encoding
        config (config_dict): source of truth
        device (torch.device): _description_

    Returns:
        str: generated text including the prompt
    """
    # if empty prompt, begin with SOS token
    if prompt == "":
        prompt = "<|endofprompt|>"
    path = pathlib.Path(config.checkpt_dir) / "checkpoint.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device)["model"])
    generated_tokens = model.generate(
        torch.tensor([encoding.encode(prompt)]).to(device), FLAGS.gen_len
    )
    # prune padding
    print(generated_tokens)
    generated_tokens = generated_tokens[generated_tokens != -1]
    return encoding.decode(list(generated_tokens))


def entrypoint(config: config_dict.ConfigDict):
    """Main entrypoint for different tasks."""
    logging.info(f"Using wandb: {config.use_wandb}")
    device = setup_device()

    # seed
    torch.manual_seed(config.seed)

    wb_config = config.to_dict()
    wb_config.update(FLAGS.flag_values_dict())

    if config.use_wandb:
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=wb_config,
        )

    # create datasets
    dataset_train = SimpleTextDataset("train", config)
    dataset_val = SimpleTextDataset("val", config)
    train_loader = DataLoader(
        dataset_train, batch_size=config.hyperparams.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset_val, batch_size=config.hyperparams.batch_size, shuffle=False
    )

    model_config = config_dict.FrozenConfigDict(
        {
            "vocab_size": dataset_train.vocab_size,
            "blocks": config.hyperparams.blocks,
            "num_heads": config.hyperparams.heads,
            "embed_dim": config.hyperparams.embed_dim,
            "block_size": config.hyperparams.block_size,
            "attn_dropout": config.hyperparams.dropout,
            "layer_dropout": config.hyperparams.dropout,
        }
    )

    # ensure loss @ init is valid
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # create model
    model = Transformer(**model_config).to(device)
    model.apply(init_weights)

    if FLAGS.run_mode == "debug":
        return init_test(model, device)
    if FLAGS.run_mode == "train":
        train(model, train_loader, val_loader, device, config)
    elif FLAGS.run_mode == "generate":
        generated_text = sample(
            model, FLAGS.prompt, dataset_train.encoding, config, device
        )
        logging.info(f"Generated text: '{generated_text}'")


def main(argv):
    del argv  # Unused.

    config = FLAGS.config
    entrypoint(config)


if __name__ == "__main__":
    app.run(main)
