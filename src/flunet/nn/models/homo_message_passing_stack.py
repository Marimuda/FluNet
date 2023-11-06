from typing import Tuple

import torch
from torch import nn as nn

from flunet.nn.conv.abstract_message_passing_stack import AbstractMessagePassingStack
from flunet.nn.conv.homo_message_passing_block import HomoMessagePassingBlock
from flunet.utils.types import ConfigDict


class HomoMessagePassingStack(AbstractMessagePassingStack):
    r"""A stack of message passing modules for homogeneous graphs.

    This module internally stacks multiple instances of `HomoMessagePassingBlock`
    and performs message passing on both node and edge features.

    Args:
        base_config (ConfigDict): Dictionary specifying the configuration of the GNN base.
            - num_blocks (int): Number of blocks in the stack.
            - use_residual_connections (bool): Whether to use residual connections.
        latent_dimension (int): Dimensionality of the internal layers.
        use_global_features (bool): Whether to use global features in the graph.
        aggregation_function_str (str): Function to use for aggregating node features.

    Shapes:
        - **Input:**
            - node_features: \((|\mathcal{V}|, F_{\text{node}})\)
            - edge_index: \((2, |\mathcal{E}|)\)
            - edge_features: \((|\mathcal{E}|, F_{\text{edge}})\)
            - global_features: \((|\mathcal{G}|, F_{\text{global}})\)
            - batch: \((|\mathcal{V}|)\)
        - **Output:** Tuple containing updated node, edge, and global features.
    """

    def __init__(
        self, base_config: ConfigDict, latent_dimension: int, use_global_features: bool, aggregation_function_str: str
    ):
        """
        Args:
            base_config: Dictionary specifying the way that the gnn base should look like.
                num_blocks: how many blocks this stack should have
                use_residual_connections: if the blocks should use residual connections. If True,
              the original inputs will be added to the outputs.
        """
        super().__init__(base_config)

        num_blocks: int = base_config.get("num_blocks")
        use_residual_connections: bool = base_config.get("use_residual_connections")

        in_global_features = latent_dimension if use_global_features else 0
        self._blocks = nn.ModuleList(
            [
                HomoMessagePassingBlock(
                    base_config=base_config,
                    in_node_features=latent_dimension,
                    in_edge_features=latent_dimension,
                    in_global_features=in_global_features,
                    aggregation_function_str=aggregation_function_str,
                    use_residual_connections=use_residual_connections,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Computes the forward pass for this homogeneous message passing stack.

        Args:
            node_features (Tensor): The features for each node of the graph.
                Shape: \((|\mathcal{V}|, F_{\text{node}})\).
            edge_index (Tensor): Connectivity Tensor of the graph.
                Shape: \((2, |\mathcal{E}|)\).
            edge_features (Tensor): Feature matrix of the edges.
                Shape: \((|\mathcal{E}|, F_{\text{edge}})\).
            global_features (Tensor): Features for the whole graph.
                Shape: \((|\mathcal{G}|, F_{\text{global}})\).
            batch (Tensor): Indexing for different graphs in the same batch.
                Shape: \((|\mathcal{V}|)\).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Updated node, edge, and global features.
        """

        for message_passing_block in self._blocks:
            node_features, edge_features, global_features = message_passing_block(
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                global_features=global_features,
                batch=batch,
            )

        output_tensors = node_features, edge_features, global_features
        return output_tensors
