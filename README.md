# 🤖 tiny-gpt-zh

[![CI](https://github.com/sworddemon85-create/tiny-gpt-zh/actions/workflows/ci.yml/badge.svg)](https://github.com/sworddemon85-create/tiny-gpt-zh/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

**从零开始训练一个会写中文的迷你 GPT。**

一个约 300 行核心代码的 Decoder-only Transformer 实现，用来理解大语言模型的完整训练流程：
字符级分词 → 数据集构造 → 因果自注意力 → 下一字符预测 → 自回归生成。
没有预训练权重、没有 Hugging Face 依赖，只有 PyTorch，CPU 就能跑。

> A minimal Chinese GPT built from scratch with pure PyTorch — train a tiny character-level language model on your laptop.

## ✨ 特性

- **极简实现** — 模型 + 分词器 + 数据集共约 300 行，每一行都可以读懂
- **GPT 标准架构** — Pre-LN 残差块、多头因果自注意力（Flash Attention 内核）、GELU 前馈层、权重共享
- **中文友好** — 字符级分词天然适合中文，词表小、无未登录词
- **CPU 可训练** — 默认配置几分钟内在笔记本 CPU 上完成训练
- **端到端测试** — CI 里包含冒烟训练测试：小模型训练 30 步，loss 必须下降，否则测试不通过
- **一键生成** — 支持温度、top-k 采样的续写脚本

## 🚀 快速开始

```bash
git clone https://github.com/sworddemon85-create/tiny-gpt-zh.git
cd tiny-gpt-zh
pip install -r requirements.txt   # torch（CPU 版即可）+ pytest

# 训练（默认配置：4 层、128 维、上下文 128 字符，CPU 几分钟）
python scripts/train.py

# 用训练好的模型续写
python scripts/sample.py --prompt "从前有一座山" --steps 200 --temperature 0.8 --top-k 10
```

## 🧠 模型结构

```
输入字符序列 (B, T)
      │
      ▼
Token 嵌入 + 位置嵌入
      │
      ▼
┌─────────────────────────────┐
│  N × Transformer Block      │
│  ├─ LayerNorm               │
│  ├─ 因果多头自注意力 (Flash) │   每个位置只能看到自己和之前
│  ├─ 残差连接                 │
│  ├─ LayerNorm               │
│  ├─ MLP（4× 升维 + GELU）    │
│  └─ 残差连接                 │
└─────────────────────────────┘
      │
      ▼
LayerNorm → 输出投影（与 Token 嵌入共享权重）
      │
      ▼
下一字符概率分布 → 交叉熵 loss
```

## 📖 学习路径

建议按这个顺序读代码，正好对应 LLM 训练的全部关键概念：

| 顺序 | 文件 | 学到什么 |
|------|------|----------|
| 1 | `tiny_gpt/tokenizer.py` | 字符级分词：文本 ↔ id 序列，词表构建 |
| 2 | `tiny_gpt/dataset.py` | 训练样本构造：x 是上下文，y 是右移一位的标签 |
| 3 | `tiny_gpt/model.py` | Transformer：因果掩码、多头注意力、残差、权重共享 |
| 4 | `scripts/train.py` | 训练循环：AdamW、梯度裁剪、train/val 划分、checkpoint |
| 5 | `scripts/sample.py` | 自回归生成：温度采样、top-k 截断 |

## 🧪 测试

```bash
pytest tests/ -v
```

- `test_tokenizer.py` — 编解码往返、词表唯一性、持久化
- `test_model.py` — 输出形状、**因果性验证**（改变未来 token 不影响过去位置的输出）、生成不越界
- `test_training.py` — **端到端冒烟训练**：极小模型训练 30 步，断言 loss 下降；训练后生成合法序列

## 📁 项目结构

```
tiny-gpt-zh/
├── tiny_gpt/
│   ├── model.py          # GPT 模型（Decoder-only Transformer）
│   ├── tokenizer.py      # 字符级分词器
│   └── dataset.py        # 训练数据集
├── scripts/
│   ├── train.py          # 训练入口（CPU 可跑）
│   └── sample.py         # 文本生成
├── data/corpus.txt       # 示例中文语料（约 5000 字）
├── tests/                # 单元测试 + 冒烟训练测试
└── .github/workflows/    # CI：安装 CPU 版 torch 并跑全部测试
```

## 🔧 换上你自己的语料

用任何 UTF-8 文本替换 `data/corpus.txt`（小说、歌词、聊天记录都可以），然后：

```bash
python scripts/train.py --data 你的语料.txt --n-layer 6 --n-embd 256 --max-iters 10000
```

语料越大、模型越大，生成效果越好。有 GPU 的话把 `--batch-size` 调大即可自动使用 CUDA。

## ⚠️ 预期效果说明

这是一个**教学项目**：默认配置下模型只有约 90 万参数、语料约 5000 字，
生成的文本会呈现出语料的句式和常用词，但不会通顺到"像人写的"。
它的价值在于把训练大模型的每一步都跑通、看懂。想要质变，请加大语料和参数量，
或参考 [nanoGPT](https://github.com/karpathy/nanoGPT) 做更大规模的训练。

## 🙏 鸣谢

架构与命名受 [nanoGPT](https://github.com/karpathy/nanoGPT) 与 [minGPT](https://github.com/karpathy/minGPT) 启发，但所有代码为独立实现，注释为中文，便于中文读者学习。

## 📄 许可证

[MIT](LICENSE)
