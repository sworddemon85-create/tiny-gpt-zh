"""字符级分词器：按字符切分文本，构建词表，支持 JSON 持久化。

对迷你语料来说字符级分词足够简单可靠——词表小、无未登录词问题、
编解码严格可逆，是理解 LLM 训练流程的最佳起点。
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        if len(set(vocab)) != len(vocab):
            raise ValueError("词表中存在重复字符")
        self.itos: list[str] = list(vocab)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """从文本构建词表（字符按字典序排列，保证可复现）。"""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """文本 -> id 序列。词表外的字符会被跳过。"""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        """id 序列 -> 文本。"""
        out = []
        for i in ids:
            if not 0 <= i < len(self.itos):
                raise IndexError(f"id {i} 超出词表范围 (0, {len(self.itos) - 1})")
            out.append(self.itos[i])
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["itos"])
