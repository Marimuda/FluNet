from typing import Dict

from torch import nn as nn

from flunet.nn.models.homo_gnn_base import GNNBase
from flunet.utils.types import ConfigDict

# Analysis of the structural changes
#   This is equivalent to the EncodeProcessDecode in literature within this domain.
#   Within quantum chemistry it is embedding -> Interaction -> Decode
#   Hence, the GNNBase has to be refactored since it contains both the Encode and Process logic
#   TODO: Extract the Encode from the GNNBase
#   If I want to follow a similar approach to the current setup, it could be done like its done at self.mlp_decoder
#   Even tho, I don't like its explicit nameo, but the params could be taken from configs.


class HomoGNN(nn.Module):
    """Homogeneous Graph Neural Network (GNN) module to process the common graph including the mesh and point cloud.

    It uses the encoder and message passing stack of the GNNBase with a node-wise decoder on top.
    """

    def __init__(
        self,
        in_node_features: Dict[str, int],
        in_edge_features: Dict[str, int],
        in_global_features: int,
        out_node_features: int,
        network_config: ConfigDict,
    ):
        """
        Args:
            in_node_features: Dictionary, where key is the node_type and values is the number of input node features in type
            in_edge_features: Dictionary, where key is the edge_types and values is the number of input edge features in type
            in_global_features: Number of input global features per graph
            out_node_features: Number of output node features (2 for 2D positions, 3 for 3D)
            network_config: Config containing information on how to build and train the overall network. Includes a config for the GNNBase.
              latent_dimension: how large the latent-dimension of the embedding should be
        """
        super().__init__()

        # HACK: I need access to the dims of node,edge,output features for normalization in LitModule
        # To not add the same config settings several times, I add it here to access it from parent.

        self.node_feat_dim = list(in_node_features.values())[0]
        self.edge_feat_dim = list(in_edge_features.values())[0]
        self.out_node_feat_dim = out_node_features

        latent_dimension = network_config.get("latent_dimension")
        self.mlp_decoder = network_config.get("mlp_decoder")

        self._gnn_base = GNNBase(
            in_node_features=in_node_features,
            in_edge_features=in_edge_features,
            in_global_features=in_global_features,
            network_config=network_config,
        )

        # define decoder
        if self.mlp_decoder:
            self.node_decoder1 = nn.Linear(latent_dimension, latent_dimension)
            self.node_decoder2 = nn.Linear(latent_dimension, out_node_features)
            self.activation = nn.LeakyReLU()
        else:
            self.node_decoder = nn.Linear(latent_dimension, out_node_features)

    def forward(self, edge_index, node_features, edge_features, global_features=None):
        """
        Performs a forward pass through the Full Graph Neural Network for the given input batch of homogeneous graphs
        Args:
            tensor: Batch of Data objects of pytorch geometric. Represents a number of homogeneous graphs
        Returns:
            Tuple.
            node_features: Decoded features of the nodse
            edge_features: Latent edges features of the last MP-Block
            global_features: Last latent global feature:q
        """
        node_features, edge_features, global_features = self._gnn_base(edge_index, node_features, edge_features)
        if self.mlp_decoder:
            node_features = self.node_decoder1(node_features)
            node_features = self.activation(node_features)
            node_features = self.node_decoder2(node_features)
        else:
            node_features = self.node_decoder(node_features)

        return node_features, edge_features, global_features


# TODO: Add HETRO MODEL
