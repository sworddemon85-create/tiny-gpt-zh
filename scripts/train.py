"""训练入口：在中文语料上从零训练 tiny-gpt-zh。

CPU 即可运行默认配置，例如：
    python scripts/train.py --n_layer 4 --n_embd 128 --max_iters 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 保证脚本可以从仓库根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_gpt.dataset import CharDataset
from tiny_gpt.model import GPT, GPTConfig
from tiny_gpt.tokenizer import CharTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, loader, device, iters: int) -> float:
    """在验证集上估算平均 loss。"""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 tiny-gpt-zh")
    parser.add_argument("--data", default="data/corpus.txt", help="语料文件路径")
    parser.add_argument("--out-dir", default="checkpoints", help="checkpoint 保存目录")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"设备: {device}")

    # 1) 读取语料并构建字符级分词器
    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    print(f"语料 {len(text)} 字符，词表 {tokenizer.vocab_size}")

    # 2) 划分训练/验证集（前 90% 训练）
    ids = tokenizer.encode(text)
    split = int(len(ids) * 0.9)
    train_set = CharDataset(ids[:split], args.block_size)
    val_set = CharDataset(ids[split:], args.block_size)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # 3) 构建模型与优化器
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"参数量: {model.num_params() / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4) 训练循环
    step = 0
    while step < args.max_iters:
        for x, y in train_loader:
            if step >= args.max_iters:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.eval_interval == 0 or step == args.max_iters - 1:
                val_loss = estimate_loss(model, val_loader, device, args.eval_iters)
                print(
                    f"step {step:5d} | train loss {loss.item():.4f} "
                    f"| val loss {val_loss:.4f}"
                )
            step += 1

    # 5) 保存 checkpoint（模型权重 + 配置 + 词表，缺一不可）
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "tiny-gpt.pt"
    torch.save(
        {
            "config": vars(config),
            "model_state": model.state_dict(),
            "itos": tokenizer.itos,
            "step": step,
        },
        ckpt_path,
    )
    print(f"checkpoint 已保存: {ckpt_path}")


if __name__ == "__main__":
    main()
