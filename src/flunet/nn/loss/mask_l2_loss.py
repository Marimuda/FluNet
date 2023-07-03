import torch
from torch.nn.modules.loss import _Loss
from einops import rearrange

from flunet.utils.common import NodeType


class L2Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(L2Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, prediction, graph, target_normalized):
        node_type = rearrange(graph.node_type, "n 1 -> n")
        loss_mask = torch.logical_or(torch.eq(node_type, NodeType.NORMAL), torch.eq(node_type, NodeType.OUTFLOW))
        error = (target_normalized - prediction) ** 2

        if self.reduction == 'none':
            loss = error[loss_mask]
        elif self.reduction == 'mean':
            loss = torch.mean(error[loss_mask])
        else:  # 'sum'
            loss = torch.sum(error[loss_mask])
        return loss
