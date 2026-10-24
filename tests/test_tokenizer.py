"""CharTokenizer 的单元测试。"""

import pytest

from tiny_gpt.tokenizer import CharTokenizer


def test_roundtrip():
    tok = CharTokenizer.from_text("你好，世界！")
    ids = tok.encode("你好，世界！")
    assert ids == [tok.stoi[ch] for ch in "你好，世界！"]
    assert tok.decode(ids) == "你好，世界！"


def test_vocab_is_sorted_and_unique():
    tok = CharTokenizer.from_text("banana")
    assert tok.itos == sorted(set("banana"))
    assert tok.vocab_size == 3  # a, b, n


def test_unknown_chars_are_skipped():
    tok = CharTokenizer.from_text("abc")
    ids = tok.encode("aXbYcZ")
    assert ids == [tok.stoi["a"], tok.stoi["b"], tok.stoi["c"]]


def test_out_of_range_id_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(IndexError):
        tok.decode([99])


def test_duplicate_vocab_raises():
    with pytest.raises(ValueError):
        CharTokenizer(["a", "a"])


def test_save_and_load(tmp_path):
    tok = CharTokenizer.from_text("春天来了，小河解冻了。")
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.itos == tok.itos
    text = "小河解冻了。"
    assert loaded.decode(loaded.encode(text)) == text
