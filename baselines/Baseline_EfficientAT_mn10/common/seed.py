# -*- coding: utf-8 -*-
"""seed 고정 — Mark5_EXP utils/seed_utils.py 와 같은 규칙(2026-08-30 판)의 사본."""
import random

import numpy as np
import torch


def set_seed(seed_value=42, deterministic=False):
    """seed 는 '어떤 무작위를 쓸 것인가', deterministic 은 'GPU 가 덧셈을 어떤 순서로 할 것인가'.
    둘을 같이 쓰면 seed 마다 다른 결과가 나오되 각 seed 는 언제 돌려도 재현된다(10~20% 느려짐)."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed_value}{' (cudnn deterministic)' if deterministic else ''}")
