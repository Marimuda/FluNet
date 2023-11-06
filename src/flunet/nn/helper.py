from typing import Iterable, Optional, Union

import numpy as np
import torch
from torch import nn


class View(nn.Module):
    """Custom Layer to change the shape of an incoming tensor."""

    def __init__(self, default_shape: Union[tuple, int], custom_repr: str = None):
        """Utility layer to reshape an input into the given shape.

        Auto-converts for different batch_sizes
        Args:
            default_shape: The shape (without the batch size) to convert to
            custom_repr: Custom representation string of this layer.
            Shown when printing the layer or a network containing it
        """

        super().__init__()
        if isinstance(default_shape, (int, np.int32, np.int64)):
            self._default_shape = (default_shape,)
        elif isinstance(default_shape, tuple):
            self._default_shape = default_shape
        else:
            raise ValueError(f"Unknown type for 'shape' parameter of View module: {default_shape}")
        self._custom_repr = custom_repr

    def forward(self, tensor: torch.Tensor, shape: Optional[Iterable] = None) -> torch.Tensor:
        """
        Shape the given input tensor with the provided shape, or with the default_shape if None is provided
        Args:
            tensor: Input tensor of arbitrary shape
            shape: The shape to fit the input into

        Returns: Same tensor, but shaped according to the shape parameter (if provided) or self._default_shape)

        """
        tensor = tensor.contiguous()  # to have everything nicely arranged in memory, which is a requisite for .view()
        if shape is None:
            return tensor.view((-1, *self._default_shape))  # -1 to deal with different batch sizes
        return tensor.view((-1, *shape))

    def extra_repr(self) -> str:
        """
        To be printed when using print(self) on either this class or some module implementing it
        Returns: A string containing information about the parameters of this module
        """
        if self._custom_repr is not None:
            return f"{self._custom_repr} default_shape={self._default_shape}"
        return f"default_shape={self._default_shape}"


class SaveBatchNorm1d(nn.BatchNorm1d):
    """Custom Layer to deal with Batchnorms for 1-sample inputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if batch_size == 1 or len(x.shape) == 1:
            return x

        return super().forward(input=x)


class LinearEmbedding(nn.Module):
    """Linear Embedding module.

    Essentially a learned matrix multiplication (and a bias) to make input dimension of tokens compatible with the
    "main" architecture that uses them.
    """

    def __init__(self, in_features: int, out_features: int):
        """

        Args:
            in_features: The total number of input features.
            out_features: The total number of output features.
        """
        super().__init__()
        if in_features == 0:
            # If there are no input features to embed, we instead learn constant initial values.
            self.is_constant_embedding = True
            in_features = 1
        else:
            self.is_constant_embedding = False

        self._out_features = out_features
        self.embedding_layer = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Linearly embed the input tensor. If the input dimension for this embedding is 0, we instead use a placeholder
        of a single one per batch as an input to generate valid output values.
        Args:
            tensor:

        Returns:

        """
        if self.is_constant_embedding:
            tensor = torch.ones(size=(*tensor.shape[:-1], 1))  # todo may need to add gpu support here
        return self.embedding_layer(tensor)

    @property
    def out_features(self) -> int:
        return self._out_features
