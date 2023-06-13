from torch import cat, nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum


class MLP(nn.Module):
    # MLP with LayerNorm
    def __init__(self, in_dim, out_dim=128, hidden_dim=128, hidden_layers=2, norm_type="BatchNorm1d"):
        """MLP.

        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        norm_type: normalization type; one of 'BatchNorm1d','BatchNorm2d','BatchNorm3d', or None
        """

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d"]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
