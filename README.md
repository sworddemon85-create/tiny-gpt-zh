# 🤖 tiny-gpt-zh

**[English](README.md)** | 简体中文

[![CI](https://github.com/sworddemon85-create/tiny-gpt-zh/actions/workflows/ci.yml/badge.svg)](https://github.com/sworddemon85-create/tiny-gpt-zh/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

**Train a mini GPT that writes Chinese — from scratch.**

A ~300-line decoder-only Transformer for understanding the full LLM training pipeline:
character-level tokenization → dataset construction → causal self-attention → next-token prediction → autoregressive generation.
No pretrained weights, no Hugging Face dependency — just PyTorch, trainable on a CPU.

> 纯 PyTorch 从零实现的迷你中文 GPT，注释为中文，适合中文读者学习。
> **[中文文档 →](README.zh-CN.md)**

## ✨ Features

- **Minimal by design** — model + tokenizer + dataset in ~300 readable lines
- **Standard GPT architecture** — Pre-LN residual blocks, multi-head causal self-attention (Flash Attention kernel), GELU MLP, weight tying
- **Chinese-friendly** — character-level tokenization suits Chinese naturally: small vocab, no OOV problem
- **CPU-trainable** — default config finishes in minutes on a laptop CPU
- **End-to-end smoke test** — CI trains a tiny model for 60 steps and asserts the loss actually decreases
- **One-command generation** — temperature and top-k sampling built in

## 🚀 Quick start

```bash
git clone https://github.com/sworddemon85-create/tiny-gpt-zh.git
cd tiny-gpt-zh
pip install -r requirements.txt   # torch (CPU build is fine) + pytest

# train (defaults: 4 layers, 128 dims, 128-char context — a few minutes on CPU)
python scripts/train.py

# generate text
python scripts/sample.py --prompt "从前有一座山" --steps 200 --temperature 0.8 --top-k 10
```

## 🧠 Architecture

```
Character sequence (B, T)
      │
      ▼
Token embedding + positional embedding
      │
      ▼
┌─────────────────────────────┐
│  N × Transformer Block      │
│  ├─ LayerNorm               │
│  ├─ causal multi-head attn  │   each position sees only its past
│  ├─ residual                │
│  ├─ LayerNorm               │
│  ├─ MLP (4× expand + GELU)  │
│  └─ residual                │
└─────────────────────────────┘
      │
      ▼
LayerNorm → output projection (weights tied to token embedding)
      │
      ▼
next-character distribution → cross-entropy loss
```

## 📖 Suggested reading order

| Step | File | What you learn |
|------|------|----------------|
| 1 | `tiny_gpt/tokenizer.py` | char-level tokenization: text ↔ ids, vocab building |
| 2 | `tiny_gpt/dataset.py` | training pairs: context x, target = x shifted by one |
| 3 | `tiny_gpt/model.py` | Transformer: causal mask, attention, residuals, weight tying |
| 4 | `scripts/train.py` | the loop: AdamW, grad clipping, train/val split, checkpoints |
| 5 | `scripts/sample.py` | autoregressive generation: temperature, top-k |

## 🧪 Tests

```bash
pytest tests/ -v
```

- `test_tokenizer.py` — roundtrip, vocab uniqueness, persistence
- `test_model.py` — shapes, **causality check** (changing future tokens doesn't affect past logits), bounded generation
- `test_training.py` — **end-to-end smoke training**: a tiny model trains for 60 steps; the test fails unless the loss drops

## 📁 Structure

```
tiny-gpt-zh/
├── tiny_gpt/            # model, tokenizer, dataset
├── scripts/             # train.py / sample.py
├── data/corpus.txt      # sample Chinese corpus (~2,500 chars, original)
├── tests/               # unit tests + smoke training test
└── .github/workflows/   # CI: CPU torch install + full test suite
```

## 🔧 Use your own corpus

Replace `data/corpus.txt` with any UTF-8 text (novels, lyrics, chat logs), then:

```bash
python scripts/train.py --data your-corpus.txt --n-layer 6 --n-embd 256 --max-iters 10000
```

## ⚠️ What to expect

This is a **teaching project**: ~0.9M parameters trained on ~2,500 characters produces text that mimics the corpus's sentence patterns but isn't human-quality fluency. The value is running and understanding every step of LLM training. For real scale, grow the corpus and model, or look at [nanoGPT](https://github.com/karpathy/nanoGPT).

## 🙏 Acknowledgments

Architecture inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [minGPT](https://github.com/karpathy/minGPT); all code is an independent implementation with Chinese comments.

## 📄 License

[MIT](LICENSE)
