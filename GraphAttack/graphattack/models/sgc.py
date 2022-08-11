import torch.nn as nn

from graphattack.config import Config
from graphattack.nn import SGConv
from graphattack.utils import wrapper

_EDGE_WEIGHT = Config.edge_weight


class SGC(nn.Module):
    """Simplifying Graph Convolution layer from paper `Simplifying Graph
    Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`__.

    Example
    -------
    # SGC model
    >>> model = SGC(100, 10)
    """
    
    @wrapper
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=2,
                 bias=True,
                 norm='both',
                 cached=True):
        super().__init__()

        conv = SGConv(in_feats,
                      out_feats,
                      bias=bias,
                      k=k,
                      norm=norm,
                      cached=cached)
        self.conv = conv

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, g, feat, edge_weight=None):
        if edge_weight is None:
            edge_weight = g.edata.get(_EDGE_WEIGHT, edge_weight)
        return self.conv(g, feat, edge_weight=edge_weight)

    def cache_clear(self):
        self.conv._cached_h = None
        return self
