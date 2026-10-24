"""tiny-gpt-zh 模型定义：Decoder-only Transformer（GPT 架构的最小实现）。

结构：Token/位置嵌入 -> N x (因果自注意力 + MLP，Pre-LN 残差) -> LayerNorm -> 输出投影。
仅依赖 PyTorch 2.0+（使用了 scaled_dot_product_attention）。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """模型超参数。默认值适合 CPU 冒烟训练，正式训练建议调大。"""

    vocab_size: int = 50257
    block_size: int = 128  # 上下文长度（一次能看到多少个字符）
    n_layer: int = 4  # Transformer Block 层数
    n_head: int = 4  # 注意力头数
    n_embd: int = 128  # 隐藏维度
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """多头因果自注意力：每个位置只能看到自己和之前的 token。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 必须能被 n_head 整除"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,  # 因果掩码由内核直接施加
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """前馈层：4 倍升维 -> GELU -> 降维。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN 残差块：x = x + Attn(LN(x)); x = x + MLP(LN(x))。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """最小可用的 GPT：语言建模 + 文本生成。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 输入嵌入与输出投影共享权重，显著减少参数量
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"序列长度 {T} 超过 block_size={self.config.block_size}")

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """自回归生成：idx 形状 (B, T)，返回 (B, T + max_new_tokens)。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        """参数量（共享权重只计一次）。"""
        return sum(p.numel() for p in self.parameters())
