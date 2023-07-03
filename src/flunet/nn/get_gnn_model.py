from flunet.nn.models.GNNModels import HomoGNN
from typing import Dict
from flunet.utils.types import ConfigDict


def get_gnn_model(
    in_node_features: Dict[str, int],
    in_edge_features: Dict[str, int],
    in_global_features: int,
    out_node_features: int,
    network_config: ConfigDict,
    hetero: bool = False,  # TODO: HETRO NOT IMPLEMENTED YET
):
    """
    Args:
        in_node_features: Dictionary, where key is the node_type and values is the number of input node features in type
        in_edge_features: Dictionary, where key is the edge_types and values is the number of input edge features in type
        in_global_features: Number of input global features per graph
        out_node_features: Number of output node features (2 for 2D positions, 3 for 3D)
        network_config: Config containing information on how to build and train the overall network. Includes a config for the GNNBase.
        hetero: False if heterogeneous Data is used
    """

    GNN = HomoGNN(
        in_node_features=in_node_features,
        in_edge_features=in_edge_features,
        in_global_features=in_global_features,
        out_node_features=out_node_features,
        network_config=network_config,
    )
    return GNN
