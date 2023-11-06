from typing import Optional

import torch

from flunet.nn.aggr.abstract_meta_module import AbstractMetaModule
from flunet.nn.dense.mlp import MLP
from flunet.utils.types import ConfigDict


class HomogeneousMetaModule(AbstractMetaModule):
    """Base class for the homogeneous modules used in the GNN.

    They are used for updating node-, edge-, and global features.
    """

    def __init__(
        self,
        in_features: int,
        num_types: int,
        latent_dimension: int,
        base_config: ConfigDict,
        out_features: Optional[int] = None,
        aggregation_function_str: str = "mean",
    ):
        """Initializes a homogeneous meta-module with a multi-layer perceptron (MLP) and an aggregation function.

        in_features: The input shape for the feedforward network
        latent_dimension: Dimensionality of the internal layers of the mlp
        out_features: The output dimension for the feedforward network
        base_config: Dictionary specifying the way that the gnn base should look like
        aggregation_function_str: How to aggregate over the nodes/edges/globals. Defaults to "mean" aggregation,
          which corresponds to torch_scatter.scatter_mean()
        """

        super().__init__(aggregation_function_str)
        mlp_config = base_config.get("mlp")
        self.num_types = num_types  # TODO: Not in use
        self._out_mlp = MLP(
            in_features=in_features, config=mlp_config, latent_dimension=latent_dimension, out_features=out_features
        )

    @property
    def out_features(self) -> int:
        """Size of the features the forward function returns."""
        return self._out_mlp.out_features


class EdgeModule(HomogeneousMetaModule):
    r"""This module computes updated edge features for a graph neural network. It concatenates source node features,
    target node features, edge attributes, and optionally global features (if provided), and then applies a multi-layer
    perceptron (MLP) to produce updated edge features.

    The update is computed as follows:

    .. math::
        \mathbf{e'} = MLP(\mathbf{src} \, \Vert \, \mathbf{dest} \, \Vert \, \mathbf{edge\_attr} \, \Vert \, \mathbf{u_{batch}})

    where \(\Vert\) denotes concatenation and \(MLP\) represents the applied multi-layer perceptron.
    If no global information is provided, the global feature component is omitted from the concatenation.

    Args:
        src (torch.Tensor): Source node features with shape \((\text{num\_edges}, \text{num\_node\_features})\).
        dest (torch.Tensor): Target node features with shape \((\text{num\_edges}, \text{num\_node\_features})\).
        edge_attr (torch.Tensor): Edge attributes with shape \((\text{num\_edges}, \text{num\_edge\_features})\).
        u (Optional[torch.Tensor]): Global features with shape \((\text{num\_graphs\_in\_batch}, \text{global\_feature\_dimension})\),
                                    one vector per graph in the batch.
        batch (Optional[torch.Tensor]): Batch vector with shape \((\text{num\_nodes},)\), assigning each node to a graph.

    Returns:
        torch.Tensor: Updated edge features with the same shape as edge_attr, after applying the MLP.

    Example:
        >>> edge_module = EdgeModule(...)
        >>> updated_edge_features = edge_module(src, dest, edge_attr, u, batch)
    """

    def forward(
        self,
        src: torch.Tensor,
        dest: torch.Tensor,
        edge_attr: torch.Tensor,
        u: Optional[torch.Tensor],
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute edge updates for the edges of the Module
        Args:
            src: (num_edges, num_node_features)
              Represents the source nodes of the graph(s), i.e., a node for each outgoing edge.
            dest: (num_edges, num_node_features)
              Represents the target nodes of the graph(s), i.e., a node for each incoming edge.
            edge_attr: (num_edges, num_edge_features). The features on each edge
            u: A matrix (num_graphs_in_batch, global_feature_dimension). Distributed along the graphs
            batch: A tensor [0, 0, ..., 0, 1, ..., 1, ..., num_graphs-1] that assigns a graph to each node

        Returns: An updated representation of the edge attributes

        """

        if u is None:  # no global information
            aggregated_features = torch.cat([src, dest, edge_attr], -1)
        else:  # has global information
            aggregated_features = torch.cat([src, dest, edge_attr, u[batch]], -1)

        out = self._out_mlp(aggregated_features)
        return out


class NodeModule(HomogeneousMetaModule):
    r"""NodeModule is a graph neural network layer designed for updating the features of the nodes in a graph based on
    their own features, the aggregated features of their neighboring nodes (via the edges), and optionally, the global
    graph features and batch information. It achieves this by utilizing a multi-layer perceptron (MLP) to process the
    concatenated input features.

    During the forward pass, the NodeModule aggregates the features of neighboring nodes for each node
    and concatenates these with the node's own features. If global graph features are present, they are
    integrated using the batch index, providing a global context to the node updates.

    Args:
        x (torch.Tensor): The node features with shape (num_nodes, num_node_features), representing
                          the feature vectors of all nodes in the graphs.
        edge_index (torch.Tensor): The edge indices with shape (2, num_edges), giving the sparse
                                   representation of the graph connectivity. The first row contains
                                   source node indices, and the second row contains destination node indices.
        edge_attr (torch.Tensor): The edge attributes with shape (num_edges, num_edge_features),
                                  indicating the features of each edge connecting the nodes in the graph.
        u (Optional[torch.Tensor]): The global features for each graph in the batch, with shape
                                    (num_graphs_in_batch, global_feature_dimension), used to enrich the
                                    node features with information from the entire graph.
        batch (Optional[torch.Tensor]): The batch vector with shape (num_nodes,), mapping each node to a
                                        graph in the batch. It's crucial for indexing global features in 'u'.

    Returns:
        torch.Tensor: The updated node features after applying the MLP, typically having the same number of
                      features as the input node features (num_nodes, num_node_features).

    Example:
        >>> node_module = NodeModule(...)
        >>> node_features = torch.rand(num_nodes, num_node_features)
        >>> edge_indices = torch.tensor([[source_node_indices], [target_node_indices]])
        >>> edge_attributes = torch.rand(num_edges, num_edge_features)
        >>> global_features = torch.rand(num_graphs_in_batch, global_feature_dimension)
        >>> batch_indices = torch.tensor([...])  # batch indices for nodes
        >>> updated_node_features = node_module(node_features, edge_indices, edge_attributes, global_features, batch_indices)
    """

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        Compute updates for each node feature vector as x_i' = f2(x_i, agg_j f1(e_ij, x_j), u),
        where f1 and f2 are MLPs
        Args:
            x: (num_nodes, num_node_features). Feature matrix for all nodes of the graphs
            edge_index: (2, num_edges). Sparse representation of the source and target nodes of each edge.
            edge_attr: (num_edges, num_edge_features). The features on each edge
            u: A matrix (num_graphs_in_batch, global_feature_dimension). Distributed along the graphs
            batch: A tensor [0, 0, ..., 0, 1, ..., 1, ..., num_graphs-1] that assigns a graph to each node
        Returns: An updated representation of the global features for each graph
        """
        src, des = edge_index  # split edges in source and target nodes
        # get source node features of all edges, combine with edge feature

        aggregated_neighbor_features = self._aggregation_function(edge_attr, src, dim=0, dim_size=x.size(0))
        if u is None:  # no global information
            aggregated_features = torch.cat([x, aggregated_neighbor_features], dim=1)
        else:  # has global information
            aggregated_features = torch.cat([x, aggregated_neighbor_features, u[batch]], dim=1)

        out = self._out_mlp(aggregated_features)
        return out


class GlobalModule(HomogeneousMetaModule):
    r"""GlobalModule is a component of a graph neural network that updates the global features of each graph in a batch.
    It aggregates node and edge features across each graph and optionally combines them with existing global features
    using a multi-layer perceptron (MLP).

    During the forward pass, the GlobalModule computes the aggregation of node features and edge features
    separately for each graph. These aggregated features are then concatenated with the current global
    features (if provided), and the result is passed through an MLP to generate updated global features
    for each graph.

    Args:
        x (torch.Tensor): The node features with shape (num_nodes, num_node_features), representing the
                          feature vectors for all nodes across the graphs in the batch.
        edge_index (torch.Tensor): The edge indices with shape (2, num_edges), indicating the connectivity
                                   of the graph by listing source and target nodes for each edge.
        edge_attr (torch.Tensor): The edge attributes with shape (num_edges, num_edge_features), containing
                                  the feature information for each edge in the graph.
        u (Optional[torch.Tensor]): The initial global features for each graph in the batch, with shape
                                    (num_graphs_in_batch, global_feature_dimension). These features provide
                                    a global context for the update.
        batch (Optional[torch.Tensor]): The batch vector with shape (num_nodes,), assigning each node to a
                                        graph in the batch. It is used to index and aggregate features at the
                                        graph level.

    Returns:
        torch.Tensor: The updated global features for each graph after processing through the MLP. The output
                      tensor has shape (num_graphs, latent_dim), where 'latent_dim' is the dimensionality of
                      the output features determined by the MLP.

    Example:
        >>> global_module = GlobalModule(...)
        >>> node_features = torch.rand(num_nodes, num_node_features)
        >>> edge_indices = torch.tensor([[source_node_indices], [target_node_indices]])
        >>> edge_attributes = torch.rand(num_edges, num_edge_features)
        >>> global_features = torch.rand(num_graphs_in_batch, global_feature_dimension)
        >>> batch_indices = torch.tensor([...])  # batch indices for nodes
        >>> updated_global_features = global_module(node_features, edge_indices, edge_attributes, global_features, batch_indices)
    """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: Optional[torch.Tensor],
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute updates for the global feature vector of each graph.
        u' = mlp(u, agg e, agg v)
        Args:
            x: (num_nodes, num_node_features). Feature matrix for all nodes of the graphs
            edge_index: (2, num_edges). Sparse representation of the source and target nodes of each edge.
            edge_attr: (num_edges, num_edge_features). The features on each edge
            u: A matrix (num_graphs_in_batch, global_feature_dimension). Distributed along the graphs
            batch: A tensor [0, 0, ..., 0, 1, ..., 1, ..., num_graphs-1] that assigns a graph to each node

        Returns: An updated representation of the global features for each graph. Has shape (num_graphs, latent_dim)
        """

        edge_batch = batch[edge_index[0]]
        # e.g., edge_index[0] = [0,1,1,2,1,1,0,1,1,0,||| 3,4,3,4,3,4,||| 6,5,7,5,8] -->
        # batch[edge_index[0] = [0,0,0,0,0,0,0,0,0,0,||| 1,1,1,1,1,1,||| 2,2,2,2,2]

        graphwise_node_aggregation = self._aggregation_function(x, batch, dim=0)
        graphwise_edge_aggregation = self._aggregation_function(edge_attr, edge_batch, dim=0)

        if u is None:
            aggregated_features = torch.cat([graphwise_node_aggregation, graphwise_edge_aggregation], dim=1)
        else:
            aggregated_features = torch.cat([u, graphwise_node_aggregation, graphwise_edge_aggregation], dim=1)

        out = self._out_mlp(aggregated_features)
        return out


class GlobalModuleNoUpdate(HomogeneousMetaModule):
    r"""GlobalModuleNoUpdate is a placeholder module in a graph neural network architecture which simply passes through
    the global features without performing any update. This module can be used in scenarios where global feature updates
    are not required or are performed by another component of the system.

    The forward pass of this module does not modify the input global features, making it computationally
    lightweight and serving as a no-operation (no-op) in the graph processing pipeline.

    Args:
        x (torch.Tensor): The node features with shape (num_nodes, num_node_features). While provided as
                          input, this data is not utilized in the forward pass of this module.
        edge_index (torch.Tensor): The edge indices with shape (2, num_edges), denoting the connections
                                   of the graph. Similar to `x`, this is not used in the forward pass.
        edge_attr (torch.Tensor): The edge attributes with shape (num_edges, num_edge_features). This
                                  input is also not used in the module's operation.
        u (Optional[torch.Tensor]): The global features for each graph in the batch with shape
                                    (num_graphs_in_batch, global_feature_dimension). This tensor is
                                    directly passed through without modification.
        batch (Optional[torch.Tensor]): The batch vector with shape (num_nodes,), assigning nodes to
                                        graphs in the batch. Like `x`, `edge_index`, and `edge_attr`,
                                        this is not utilized in the forward pass.

    Returns:
        torch.Tensor: The unmodified input global features, `u`, are returned as the output.

    Example:
        >>> no_update_module = GlobalModuleNoUpdate(...)
        >>> node_features = torch.rand(num_nodes, num_node_features)
        >>> edge_indices = torch.tensor([[source_node_indices], [target_node_indices]])
        >>> edge_attributes = torch.rand(num_edges, num_edge_features)
        >>> global_features = torch.rand(num_graphs_in_batch, global_feature_dimension)
        >>> batch_indices = torch.tensor([...])  # batch indices for nodes
        >>> unchanged_global_features = no_update_module(node_features, edge_indices, edge_attributes, global_features, batch_indices)
    """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: Optional[torch.Tensor],
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Returns input global features u without update."""
        return u
