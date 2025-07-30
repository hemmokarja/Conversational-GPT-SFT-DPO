import math
import inspect
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_size = config.n_embd // config.n_head

    def forward(self, x):
        B, S, C = x.size()

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, S, self.n_head, self.head_size).transpose(1, 2)  # [B, nh, S, hs]
        q = q.view(B, S, self.n_head, self.head_size).transpose(1, 2)  # [B, nh, S, hs]
        v = v.view(B, S, self.n_head, self.head_size).transpose(1, 2)  # [B, nh, S, hs]

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )  # [B, nh, S, hs]
        y = y.transpose(1, 2).contiguous().view(B, S, C)  # [B, S, C]
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        self._init_skip_proj_weights()

        logger.info(
            f"Initialized GPT with {self.get_num_params() / 1e6:.2f} M parameters "
            f"(of which {self.get_num_params('wte') / 1e6:.2f} M in embeddings)"
        )

    @torch.no_grad()
    def generate(self, x, max_tokens, temperature=1.0, top_k=None):
        for _ in range(max_tokens):

            x = x[:, -self.config.block_size:]
            logits, _ = self(x)  # [B, 1, vocab]
            logits = logits[:, -1, :] / temperature  # [B, vocab]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)  # [B, vocab]
            x_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            x = torch.cat((x, x_next), dim=1)  # [B, S + 1]

        return x

    def forward(self, x, y=None):
        B, S = x.size()
        
        if S > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {S}, block size is only "
                f"{self.config.block_size}"
            )

        tok_emb = self.transformer.wte(x)  # [B, S, C]
        pos_emb = self.transformer.wpe(torch.arange(S, device=x.device))  # [B, S, C]
        emb = tok_emb + pos_emb
        emb = self.transformer.drop(emb)
        for block in self.transformer.h:
            emb = block(emb)
        emb = self.transformer.ln_f(emb)

        if y is not None:
            logits = self.lm_head(emb)  # [B, S, vocab]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0
            )
        else:
            logits = self.lm_head(emb[:, [-1], :])  # [B, 1, vocab]
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        if model_type not in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                "Supported types are: gpt2, gpt2-medium, gpt2-large, gpt2-xl"
            )

        override_args = override_args or {}
        if not all(k == "dropout" for k in override_args):
            raise ValueError("Only dropout can be overridden")
        
        from transformers import GPT2LMHeadModel
        logger.info(f"Initializing a pre-trained {model_type} model...")

        config_args = {
            "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},  # 124M params
            "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},  # 350M params
            "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},  # 774M params
            "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},  # 1558M params
        }[model_type]

        logger.debug("Forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True

        if "dropout" in override_args:
            logger.info(f"Overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        state = model.state_dict()
        state_keys = [k for k in state.keys() if not k.endswith(".attn.bias")] # discard buffer

        logger.info("Loading pre-trained weights from HuggingFace...")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        state_hf = model_hf.state_dict()

        state_keys_hf = [
            k for k in state_hf.keys()
            if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias")) # discard buffers
        ]

        if len(state_keys_hf) != len(state_keys):
            raise ValueError(
                f"Mismatch in number of parameters: {len(state_keys_hf)} (HF) vs "
                f"{len(state_keys)}."
            )
        
        # OpenAI GPT checkpoints use a "Conv1D" module for the attention and MLP layers,
        # but we only want to use a vanilla Linear layer - thus we transpose the weights
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        for k in state_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert state_hf[k].shape[::-1] == state[k].shape
                with torch.no_grad():
                    state[k].copy_(state_hf[k].t())
            else:
                assert state_hf[k].shape == state[k].shape
                with torch.no_grad():
                    state[k].copy_(state_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Only >=2D parameters will be weight decayed, i.e. all weight tensors in
        # matmuls + embeddings decay, but biases and layernorms won't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == "cuda"
        extra_args = {"fused": True} if use_fused else {}

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

        return optimizer

    def get_num_params(self, name_contains=None):
            if name_contains is None:
                name_contains = ""
            return sum(
                p.numel() for pn, p in self.named_parameters() if name_contains in pn
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_skip_proj_weights(self):
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
                )
