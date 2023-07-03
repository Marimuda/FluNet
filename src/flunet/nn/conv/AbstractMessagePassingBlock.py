from typing import Optional
from torch import nn as nn
from torch_geometric.nn import MetaLayer

from flunet.utils.types import *


class AbstractMessagePassingBlock(nn.Module):
    """
    Defines a single MessagePassingLayer that takes an observation graph and updates its node and edge
    features using different modules described in implementations of this abstract class.
    It first updates the edge-features. The node-features are updated next using the new edge-features. Finally,
    it updates the global features using the new edge- & node-features. The updates are done through MLPs.
    The three Modules (Edge, Node, Global) are combined into a MetaLayer.
    """

    def __init__(self, base_config: ConfigDict, use_residual_connections: bool = False) -> None:
        """
        Args:
            base_config: Dictionary specifying the way that the gnn base should look like.
            use_residual_connections: Whether to use residual connections for both nodes and edges or not. If True,
                the original inputs will be added to the outputs.
        """
        super().__init__()
        self._use_residual_connections = use_residual_connections

        # structure created by specific implementation
        self._out_edge_features: int = 0
        self._out_node_features: int = 0
        self._out_global_features: int = 0
        self._meta_layer: Optional[MetaLayer] = None

    @property
    def out_node_features(self) -> int:
        """
        The output dimension of the node module
        """
        return self._out_node_features

    @property
    def out_edge_features(self) -> int:
        """
        The output dimension of the node module
        """
        return self._out_edge_features

    @property
    def out_global_features(self) -> int:
        """
        The output dimension of the global module
        """
        return self._out_global_features

    def __repr__(self):
        return f"MetaLayer: {self._meta_layer}," f"use_residual_connections: {self._use_residual_connections},"
