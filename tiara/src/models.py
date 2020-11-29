import torch
from torch import nn
import torch.nn.functional as F


# 0 - plastid
# 1 - bacteria
# 2 - mitochondria
# 3 - archeal
# 4 - eukarya


def mish(x):
    return x * torch.tanh(F.softplus(x))


class mish_layer(nn.Module):
    def __init__(self):
        super(mish_layer, self).__init__()

    def forward(self, _input):
        return mish(_input)


class NNet1(nn.Sequential):
    def __init__(self, dim_in, hidden_1, hidden_2, dim_out, dropout):
        super().__init__(
            nn.Linear(dim_in, hidden_1),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, dim_out),
            nn.Softmax(1),
        )


class NNet2(nn.Sequential):
    def __init__(self, dim_in, hidden_1, hidden_2, dim_out, dropout):
        super().__init__(
            nn.Linear(dim_in, hidden_1),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, dim_out),
            nn.Softmax(1),
        )
