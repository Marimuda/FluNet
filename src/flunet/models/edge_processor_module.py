from typing import Type

import torch
from torch import Tensor, nn
from torch_geometric.data import Data


class EdgeProcessorModule(nn.Module):
    def __init__(
        self,
        in_dim_node: int,
        in_dim_edge: int,
        hidden_dim: int,
        hidden_layers: int,
        norm_type: str = "BatchNorm1d",
        activation: Type[nn.Module] = None,
    ):
        """Initializes an EdgeProcessorModule.

        Args:
            in_dim_node (int): The input dimensionality of node features.
            in_dim_edge (int): The input dimensionality of edge features.
            hidden_dim (int): The dimensionality of hidden layers.
            hidden_layers (int): The number of hidden layers.
            norm_type (str, optional): The type of normalization layer. Defaults to "BatchNorm1d".
            activation (type[nn.Module], optional): The activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_features=in_dim_node * 2 + in_dim_edge, out_features=hidden_dim))
        layers.append(activation())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            layers.append(activation())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=in_dim_edge))

        self.layers = nn.Sequential(*layers)

        if norm_type is not None:
            assert norm_type in ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d"]
            self.norm_layer = getattr(nn, norm_type)(in_dim_edge)

    def forward(self, src: Tensor, dest: Tensor, edge_attr: Tensor, u: Tensor, batch: Tensor) -> Tensor:
        """Performs a forward pass through the EdgeProcessorModule.

        Args:
            src (Tensor): The source node features.
            dest (Tensor): The destination node features.
            edge_attr (Tensor): The edge features.
            u (Tensor): A tensor used for message passing.
            batch (Tensor): A tensor representing the graph batch.

        Returns:
            Tensor: The processed edge features.
        """
        x = torch.cat([src, dest, edge_attr], -1)
        for layer in self.layers:
            x = layer(x)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        x += edge_attr

        return x
