import torch
from torch_geometric.data import Data

class EdgeProcessorModule(torch.nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, hidden_dim, hidden_layers, norm_type='BatchNorm1d'):
        super().__init__()

        self.layers = []
        self.layers.append(torch.nn.Linear(in_features=in_dim_node * 2 + in_dim_edge, out_features=hidden_dim))
        self.layers.append(torch.nn.ReLU())

        for _ in range(hidden_layers - 1):
            self.layers.append(torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(in_features=hidden_dim, out_features=in_dim_edge))

        if norm_type is not None:
            assert norm_type in ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d']
            self.norm_layer = getattr(torch.nn, norm_type)(in_dim_edge)

    def forward(self, src, dest, edge_attr, u, batch):
        x = torch.cat([src, dest, edge_attr], -1)
        for layer in self.layers:
            x = layer(x)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        x += edge_attr

        return x
