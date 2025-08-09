import math
import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass
class GPTConfig:
    seq_len: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    padding_idx: Optional[int] = None
    ignored_idx: int = -100


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: list = None  # Which modules to apply LoRA to

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn", "c_proj", "c_fc"]


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

    def forward(self, x, attn_mask=None):
        B, S, C = x.size()

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, S, self.n_head, self.head_size).transpose(1, 2)  # [B, nh, S, hs]
        q = q.view(B, S, self.n_head, self.head_size).transpose(1, 2)  # [B, nh, S, hs]
        v = v.view(B, S, self.n_head, self.head_size).transpose(1, 2)  # [B, nh, S, hs]

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True if attn_mask is None else False
        )  # [B, nh, S, hs]
        y = y.transpose(1, 2).contiguous().view(B, S, C)  # [B, S, C]
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


def _get_causal_attn_mask(seq_len, device):
    causal_mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    )
    return causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]


def _get_from_to_attn_mask(valid_mask):
    # prevent attending *to* pad tokens
    to_mask = valid_mask.unsqueeze(1)  # [B, 1, S]
    # prevent attending *from* pad tokens (= generate outputs for padding tokens)
    from_mask = valid_mask.unsqueeze(2)  # [B, S, 1]
    from_to_mask = from_mask & to_mask  # [B, S, S]
    return from_to_mask.unsqueeze(1).bool()  # [B, 1, S, S]


def _get_combined_attn_mask(seq_len, valid_mask, device):
    """
    Build an attention mask that prevents attending to future tokens and padded tokens.
    The resulting mask has True indicating that the element should take part in
    attention. The resulting mask must be broadcastable to tensor of attention
    weights produced by the scaled dot-product attention computation, [B, nh, S, S].

    valid_mask: [B, S]
    attn_mask: [B, 1, S, S]
    """
    if valid_mask is None:
        return None
    causal_mask = _get_causal_attn_mask(seq_len, device)  # [1, 1, S, S]
    from_to_mask = _get_from_to_attn_mask(valid_mask)  # [B, 1, S, S]
    return causal_mask & from_to_mask  # [B, 1, S, S]


def _apply_token_restrictions(logits, prevent_tokens=None):
    if prevent_tokens is not None and len(prevent_tokens) > 0:
        logits[..., prevent_tokens] = float("-inf")
    return logits


class GPT2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.seq_len, config.n_embd),
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

    def forward(self, x, y=None, valid_mask=None, prevent_tokens=None):
        """
        x: [B, S] tensor of token indices
        y: [B, S] tensor of target token indices (optional, for training)
        valid_mask: [B, S]  True = valid token, False = padding
        """
        B, S = x.size()

        if S > self.config.seq_len:
            raise ValueError(
                f"Cannot forward sequence of length {S}, block size is only "
                f"{self.config.seq_len}"
            )
        
        attn_mask = _get_combined_attn_mask(S, valid_mask, x.device)

        tok_emb = self.transformer.wte(x)  # [B, S, C]
        pos_emb = self.transformer.wpe(torch.arange(S, device=x.device))  # [B, S, C]
        emb = tok_emb + pos_emb

        emb = self.transformer.drop(emb)
        for block in self.transformer.h:
            emb = block(emb, attn_mask)
        emb = self.transformer.ln_f(emb)

        if y is not None:
            logits = self.lm_head(emb)  # [B, S, vocab]
            logits = _apply_token_restrictions(logits, prevent_tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=self.config.ignored_idx
            )
        else:
            logits = self.lm_head(emb[:, [-1], :])  # [B, 1, vocab]
            logits = _apply_token_restrictions(logits, prevent_tokens)
            loss = None

        return logits, loss

    @staticmethod
    def _should_stop(x, end_tokens):
        n = len(end_tokens)
        return x.view(-1)[-n:].cpu().tolist() == end_tokens

    @torch.no_grad()
    def generate(
        self,
        x,
        max_tokens,
        temperature=1.0,
        top_k=None,
        end_tokens=None,
        prevent_tokens=None,
    ):
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor (B, S), got {x.dim()}D")
        if x.size(0) > 1:
            raise ValueError("Batched generation not supported")

        for _ in range(max_tokens):

            x = x[:, -self.config.seq_len:]
            logits, _ = self(x, prevent_tokens=prevent_tokens)  # [B, 1, vocab]
            logits = logits[:, -1, :] / temperature  # [B, vocab]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)  # [B, vocab]
            x_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            x = torch.cat((x, x_next), dim=1)  # [B, S + 1]

            if end_tokens is not None and self._should_stop(x, end_tokens):
                break

        return x

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        supported_configs = {
            "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},  # 124M params
            "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},  # 350M params
            "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},  # 774M params
            "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},  # 1558M params
        }
        if model_type not in supported_configs:
            raise ValueError(
                f"Unsupported model type: {model_type}. Supported types are: "
                f"{', '.join(supported_configs.keys())}"
            )

        override_args = override_args or {}
        allowed_overrides = {"dropout"}
        if not all(k in allowed_overrides for k in override_args):
            raise ValueError(f"Allowed overrides are: {', '.join(allowed_overrides)}")
        
        logger.info(f"Initializing a pre-trained {model_type} model...")

        config_args = supported_configs[model_type]
        logger.debug("Forcing vocab_size=50257, seq_len=1024, bias=True")  # per gpt2
        config_args["vocab_size"] = 50257
        config_args["seq_len"] = 1024
        config_args["bias"] = True

        for arg_name, arg_value in override_args.items():
            logger.info(f"Overriding {arg_name} to {arg_value}")
            config_args[arg_name] = arg_value

        config = GPTConfig(**config_args)
        model = cls(config)
        state = model.state_dict()
        state_keys = [k for k in state.keys() if not k.endswith(".attn.bias")] # discard buffer

        logger.info("Loading pre-trained weights from HuggingFace...")
        from transformers import GPT2LMHeadModel
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

    def to_fine_tuneable(self):
        return FineTuneableGPT2.from_base_model(self)


class LoRALinear(nn.Module):
    def __init__(
        self, in_features, out_features, r=8, alpha=32, dropout=0.0, bias=True
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r=8, alpha=32, dropout=0.0):
        if not isinstance(linear, nn.Linear):
            raise ValueError("Only nn.Linear layers can be converted to LoRALinear")

        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )
        lora.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora.bias.data.copy_(linear.bias.data)
        return lora

    def forward(self, x):
        x_base = F.linear(x, self.weight, self.bias)  # [*, out_features]

        x_lora = self.dropout(x) if self.training else x
        x_lora = F.linear(x, self.lora_A)  # [*, r]
        x_lora = F.linear(x_lora, self.lora_B) # [*, out_features]
        x_lora = x_lora * self.scale

        return x_base + x_lora


class FineTuneableGPT2(GPT2):
    def __init__(self, config):
        super().__init__(config)
        self.original_vocab_size = config.vocab_size
        self.new_token_indices = []

    @classmethod
    def from_base_model(cls, base_model):
        model = cls(base_model.config)
        model.load_state_dict(base_model.state_dict())
        return model

    def add_padding_token(self):
        vocab_size, embed_dim = self.transformer.wte.weight.shape
        new_wte = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        new_wte.weight.data[:vocab_size] = self.transformer.wte.weight.data
        self.transformer.wte = new_wte

        self.lm_head = nn.Linear(embed_dim, vocab_size + 1, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # re-tie
        
        self.config.vocab_size += 1
        self.config.padding_idx = vocab_size

    def apply_lora(self, lora_config):
        self.lora_config = lora_config

        n_trained_old = sum(p.numel() for p in self.parameters())

        lora_kwargs = {
            "r": self.lora_config.r,
            "alpha": self.lora_config.alpha,
            "dropout": self.lora_config.dropout,
        }
        for block in self.transformer.h:
            if "c_attn" in self.lora_config.target_modules:
                block.attn.c_attn = LoRALinear.from_linear(
                    block.attn.c_attn, **lora_kwargs
                )
            if "c_proj" in self.lora_config.target_modules:
                block.attn.c_proj = LoRALinear.from_linear(
                    block.attn.c_proj, **lora_kwargs
                )
                block.mlp.c_proj = LoRALinear.from_linear(
                    block.mlp.c_proj, **lora_kwargs
                )
            if "c_fc" in self.lora_config.target_modules:
                block.mlp.c_fc = LoRALinear.from_linear(
                    block.mlp.c_fc, **lora_kwargs
                )

        logger.info(f"Initialized LoRA layers for modules: {self.lora_config.target_modules}")

        self._freeze_parameters()

        n_trained = sum(p.numel() for p in self.get_optimizer_parameters())

        logger.info(
            f"LoRA initialized: num. of parameters requiring gradient computation: "
            f"{n_trained_old / 1e6:.2f} M -> {n_trained / 1e6:.2f} M"
        )

    def _freeze_parameters(self):
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_optimizer_parameters(self):
        """
        Get parameters that should be passed to the optimizer.

        Returns parameters where requires_grad=True. Note that for embeddings,
        even though requires_grad=True, the backward hook ensures only new tokens
        receive gradient updates.
        """
        return [param for param in self.parameters() if param.requires_grad]
