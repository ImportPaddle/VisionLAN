import numpy as np
import os
import sys
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
import torch
import paddle

SEED = 100
torch.manual_seed(SEED)
paddle.seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':
    pass
