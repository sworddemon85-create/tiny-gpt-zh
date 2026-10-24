"""GPT 模型的单元测试：形状、因果性、生成。"""

import pytest
import torch

from tiny_gpt.model import GPT, GPTConfig


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=50,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    model.eval()
    return model


def test_forward_logits_shape(small_model):
    idx = torch.randint(0, 50, (2, 16))
    logits, _ = small_model(idx)
    assert logits.shape == (2, 16, 50)


def test_loss_is_scalar_and_finite(small_model):
    idx = torch.randint(0, 50, (2, 16))
    targets = torch.randint(0, 50, (2, 16))
    _, loss = small_model(idx, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_sequence_longer_than_block_size_raises(small_model):
    idx = torch.randint(0, 50, (1, 33))  # block_size=32
    with pytest.raises(ValueError):
        small_model(idx)


def test_causal_masking(small_model):
    """改变未来的 token 不应影响过去位置的 logits。"""
    torch.manual_seed(42)
    prefix = torch.randint(0, 50, (1, 12))
    a = torch.cat([prefix, torch.tensor([[7]])], dim=1)
    b = torch.cat([prefix, torch.tensor([[42]])], dim=1)

    with torch.no_grad():
        logits_a, _ = small_model(a)
        logits_b, _ = small_model(b)

    # 前 12 个位置的输出必须完全一致（只有最后一个位置由未来决定）
    assert torch.allclose(logits_a[:, :12], logits_b[:, :12], atol=1e-5)
    # 最后一个位置应当不同
    assert not torch.allclose(logits_a[:, 12], logits_b[:, 12], atol=1e-5)


def test_generate_shape(small_model):
    idx = torch.randint(0, 50, (2, 8))
    out = small_model.generate(idx, max_new_tokens=10, top_k=5)
    assert out.shape == (2, 18)
    # 生成不越界
    assert out.min() >= 0 and out.max() < 50


def test_num_params_positive(small_model):
    assert small_model.num_params() > 0


def test_embedding_tied_with_lm_head(small_model):
    """输入嵌入与输出投影共享同一份权重张量。"""
    assert small_model.tok_emb.weight.data_ptr() == small_model.lm_head.weight.data_ptr()
