"""训练数据集：把 id 序列切成固定长度片段，构造下一字符预测样本。

每个样本是 (x, y) 对，其中 y 是 x 右移一位的结果——这正是语言模型的训练目标。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, ids: list[int], block_size: int):
        if len(ids) < block_size + 1:
            raise ValueError(
                f"语料太短：共 {len(ids)} 个字符，至少需要 block_size+1 = {block_size + 1} 个"
            )
        self.ids = ids
        self.block_size = block_size

    def __len__(self) -> int:
        # 每个合法起点对应一个样本
        return len(self.ids) - self.block_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.tensor(self.ids[i : i + self.block_size + 1], dtype=torch.long)
        return chunk[:-1], chunk[1:]
