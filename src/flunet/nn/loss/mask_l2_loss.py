import torch
from einops import rearrange
from torch.nn.modules.loss import _Loss

from flunet.utils.common import NodeType


class L2Loss(_Loss):
    """The L2Loss class extends PyTorch's loss modules to calculate the L2 loss (mean squared error) for graph-based
    data, considering only specific node types in the computation.

    By applying a mask that selects either NORMAL or OUTFLOW node types, it ensures that only the
    relevant nodes contribute to the loss calculation. This can be useful in scenarios where
    different nodes have different roles or significance in the loss computation.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default, the losses
                                       are averaged over each loss element in the batch.
        reduce (bool, optional): Deprecated (see :attr:`reduction`).
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                   'none': no reduction will be applied.
                                   'mean': the sum of the output will be divided by the number of elements
                                           in the output.
                                   'sum': the output will be summed.

    The 'forward' method calculates the L2 loss between `prediction` and `target_normalized`, considering
    only the nodes that match the specified `NodeType`.

    Args:
        prediction (torch.Tensor): The predicted values.
        graph (Graph): The graph object containing node types.
        target_normalized (torch.Tensor): The normalized target values.

    Returns:
        torch.Tensor: The calculated L2 loss, which can be either the sum or the mean of the losses,
                      depending on the 'reduction' parameter.

    Raises:
        ValueError: If the 'reduction' parameter is not one of 'none', 'mean', or 'sum'.
    """

    __constants__ = ["reduction"]

    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)

    def forward(self, prediction, graph, target_normalized):
        node_type = rearrange(graph.node_type, "n 1 -> n")
        loss_mask = torch.logical_or(torch.eq(node_type, NodeType.NORMAL), torch.eq(node_type, NodeType.OUTFLOW))
        error = (target_normalized - prediction) ** 2

        if self.reduction == "none":
            loss = error[loss_mask]
        elif self.reduction == "mean":
            loss = torch.mean(error[loss_mask])
        else:  # 'sum'
            loss = torch.sum(error[loss_mask])
        return loss
