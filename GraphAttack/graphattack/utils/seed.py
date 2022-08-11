import random
from numbers import Number
from typing import Optional

import dgl
import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: Optional[int] = None):
    assert seed is None or isinstance(seed, Number), seed
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        dgl.random.seed(seed)
        # torch.cuda.manual_seed_all(seed)
