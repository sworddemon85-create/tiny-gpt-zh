"""冒烟训练测试：极小模型在合成语料上训练 30 步，loss 必须下降。

这是整个训练管线的端到端验证——数据、模型、优化器任何一环出问题，
这条测试都会红。CPU 上几秒钟即可完成。
"""

import torch
from torch.utils.data import DataLoader

from tiny_gpt.dataset import CharDataset
from tiny_gpt.model import GPT, GPTConfig
from tiny_gpt.tokenizer import CharTokenizer

CORPUS = (
    "从前有一座山，山里有一个小村庄。村里有一条小河，河水清清的。"
    "小河边有一棵老树，树上住着一只小鸟。小鸟每天早上都唱歌。"
    "太阳升起来了，新的一天开始了。村里的人都说：早上好！"
    "秋天的时候，小狐狸把红红的果子一颗一颗捡回家。冬天来了，"
    "外面下起了雪，小狐狸在家里吃果子，一点也不用发愁。"
) * 6  # 重复若干遍，保证样本数量足够


def test_smoke_training_loss_decreases():
    torch.manual_seed(1337)

    tokenizer = CharTokenizer.from_text(CORPUS)
    ids = tokenizer.encode(CORPUS)
    assert len(ids) > 64, "合成语料太短，测试配置错误"

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    dataset = CharDataset(ids, block_size=32)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    model.train()
    step = 0
    while step < 30:
        for x, y in loader:
            if step >= 30:
                break
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1

    # 用前 5 步与后 5 步的平均值比较，避免单步抖动造成误判
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early * 0.8, (
        f"训练 30 步后 loss 未明显下降: initial={early:.4f}, final={late:.4f}"
    )


def test_smoke_generation_after_training():
    torch.manual_seed(1337)
    tokenizer = CharTokenizer.from_text(CORPUS)
    ids = tokenizer.encode(CORPUS)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    dataset = CharDataset(ids, block_size=32)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    step = 0
    while step < 30:
        for x, y in loader:
            if step >= 30:
                break
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

    # 训练后的模型应能正常生成合法 id 序列
    model.eval()
    idx = torch.tensor([ids[:8]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=32, temperature=0.8, top_k=10)
    assert out.shape == (1, 40)
    assert out.min() >= 0 and out.max() < tokenizer.vocab_size
