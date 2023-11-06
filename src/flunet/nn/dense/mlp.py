from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn as nn
from torch.nn import utils

from flunet.nn.helper import View
from flunet.nn.utility import (
    get_activation_and_regularization_layers,
    get_layer_size_layout,
)
from flunet.utils.types import ConfigDict, Shape

# define a dictionary that maps activation function names to functions
activation_functions = {
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(),
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
}


def build_mlp(
    in_features: Shape,
    feedforward_config: ConfigDict,
    latent_dimension: Optional[int] = None,
    out_features: Optional[int] = None,
) -> Tuple[nn.ModuleList, int]:
    """Builds the discriminator (sub)network.

    This part of the network accepts some latent space as the input and
    outputs a classification
    Args:
        in_features: Number of input features
        feedforward_config: Dictionary containing the specification for the feedforward network.
        Includes num_layers, max_neurons and a shape of the network. May also include a subdict "regularization"
        for dropout, batchnorm etc.
        latent_dimension: Optional latent size of the mlp layers.
        Overwrites "max_neurons" and "network_shape if provided.
        out_features: Output dimension of the feedforward network
    Returns: A nn.ModuleList representing the discriminator module, and the size of the final activation of this module
    """

    forward_layers = nn.ModuleList()
    if isinstance(in_features, (int, np.int32, np.int64)):  # can use in_features directly
        pass
    elif isinstance(in_features, (tuple)):
        if len(in_features) == 1:  # Only one dimension can be used
            in_features = in_features[0]
        else:  # More than one dimension, need to flatten first
            in_features: int = int(np.prod(in_features))
            forward_layers.append(View(default_shape=(in_features,), custom_repr="Flattening feedforward input to 1d"))
    else:
        raise ValueError(f"Unknown type for in_features: {type(in_features)} parameter in mlp.py")

    activation_function: str = feedforward_config.get("activation_function").lower()
    regularization_config = feedforward_config.get("regularization", {})
    spectral_norm: bool = regularization_config.get("spectral_norm", False)
    output_layer: bool = feedforward_config.get("output_layer")

    previous_shape = in_features

    if latent_dimension is None:
        network_layout = get_layer_size_layout(
            max_neurons=feedforward_config.get("max_neurons"),
            num_layers=feedforward_config.get("num_layers"),
            network_shape=feedforward_config.get("network_shape"),
        )
    else:
        network_layout = np.repeat(latent_dimension, feedforward_config.get("num_layers"))

    for current_layer_size in network_layout:
        # add main linear layer
        if spectral_norm:
            forward_layers.append(
                utils.spectral_norm(nn.Linear(in_features=previous_shape, out_features=current_layer_size))
            )
        else:
            forward_layers.append(nn.Linear(in_features=previous_shape, out_features=current_layer_size))

        # add activation function
        activation_fn = activation_functions.get(activation_function, None)
        if activation_fn is None:
            raise ValueError(f"Unknown activation function '{activation_function}'")
        forward_layers.append(activation_fn)

        additional_layers = get_activation_and_regularization_layers(
            in_features=current_layer_size, regularization_config=regularization_config
        )

        forward_layers.extend(additional_layers)
        previous_shape = current_layer_size

    if out_features is not None:  # linear embedding to output size
        forward_layers.append(nn.Linear(in_features=previous_shape, out_features=out_features))
    else:
        out_features = previous_shape
        if output_layer:
            out_features = latent_dimension
            forward_layers.append(nn.Linear(in_features=previous_shape, out_features=out_features))

    return forward_layers, out_features


class MLP(nn.Module):
    # Feedforward module with LayerNorm. Gets some input x and computes an output f(x).
    def __init__(
        self,
        in_features: Union[tuple, int],
        config: ConfigDict,
        latent_dimension: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        """
        Initializes a new encoder module that consists of multiple different layers that together encode some
        input into a latent space
        Args:
            in_features: The input shape for the feedforward network
            out_features: The output dimension for the feedforward network
            latent_dimension: Optional latent size of the mlp layers.
                Overwrites "max_neurons" and "network_shape if provided.
            config: Dict containing information about what kind of feedforward network to build as well
              as how to regularize it (via batchnorm etc.)
        """

        super().__init__()
        self.feedforward_layers, self._out_features = build_mlp(
            in_features=in_features,
            feedforward_config=config,
            latent_dimension=latent_dimension,
            out_features=out_features,
        )

    @property
    def out_features(self) -> int:
        """
        Returns the output dimension of the feedforward network
        Returns:
        """
        return self._out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feedforward network.

        Args:
            x: Input torch.Tensor
        Returns:
            Output torch.Tensor after passing through the feedforward network f(x)
        """
        for layer in self.feedforward_layers:
            x = layer(x)
        return x
