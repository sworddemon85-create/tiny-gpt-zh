"""文本生成：加载 checkpoint，续写一段中文。

例如：
    python scripts/sample.py --prompt "从前有一座山" --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_gpt.model import GPT, GPTConfig
from tiny_gpt.tokenizer import CharTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="tiny-gpt-zh 文本生成")
    parser.add_argument("--ckpt", default="checkpoints/tiny-gpt.pt", help="checkpoint 路径")
    parser.add_argument("--prompt", default="", help="起始文本（留空则从词表首字符开始）")
    parser.add_argument("--steps", type=int, default=200, help="生成多少个新字符")
    parser.add_argument("--temperature", type=float, default=0.8, help="越高越随机")
    parser.add_argument("--top-k", type=int, default=None, help="只在概率前 k 的字符中采样")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    tokenizer = CharTokenizer(ckpt["itos"])
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"已加载 {args.ckpt}（训练到 step {ckpt.get('step', '?')}，设备 {device}）")

    ids = tokenizer.encode(args.prompt)
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("-" * 40)
    print(tokenizer.decode(out[0].tolist()))
    print("-" * 40)


if __name__ == "__main__":
    main()
