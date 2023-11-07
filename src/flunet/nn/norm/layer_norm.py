import logging

import torch
from torch import Tensor, nn

# Configure the logger
logging.basicConfig(level=logging.INFO)


class LayerNorm(nn.Module):
    """A custom Layer Normalization module that normalizes the input tensor over the channel dimension.

    Attributes:
        dim (int): The number of features in the input tensor.
        eps (float): A value added to the denominator for numerical stability.
        g (Tensor): The learnable gain parameters.
        b (Tensor): The learnable bias parameters.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initializes the LayerNorm module.

        Args:
            dim (int): The number of features in the input tensor.
            eps (float): A value added to the denominator for numerical stability.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError("The `dim` parameter must be positive.")
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for LayerNorm.

        Args:
            x (Tensor): The input tensor to be normalized.

        Returns:
            Tensor: The normalized tensor.
        """
        if x.dim() != 4:
            raise ValueError("Expected input with 4 dimensions (batch, channels, height, width)")
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        mean = torch.mean(x, dim=1, keepdim=True)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.g + self.b


class PreNorm(nn.Module):
    """Applies Layer Normalization before a module function.

    Attributes:
        fn (nn.Module): The module function to apply after normalization.
        norm (LayerNorm): An instance of the LayerNorm module.
    """

    def __init__(self, dim: int, fn: nn.Module) -> None:
        """Initializes the PreNorm module.

        Args:
            dim (int): The number of features in the input tensor for normalization.
            fn (nn.Module): The module function to apply after normalization.
        """
        super().__init__()
        if not isinstance(fn, nn.Module):
            raise TypeError("The `fn` argument must be an instance of `nn.Module`.")
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for PreNorm.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying normalization and the module function.
        """
        x = self.norm(x)
        return self.fn(x)


class PostNorm(nn.Module):
    """Applies Layer Normalization after a module function.

    Attributes:
        fn (nn.Module): The module function to apply before normalization.
        norm (LayerNorm): An instance of the LayerNorm module.
    """

    def __init__(self, dim: int, fn: nn.Module) -> None:
        """Initializes the PostNorm module.

        Args:
            dim (int): The number of features in the input tensor for normalization.
            fn (nn.Module): The module function to apply before normalization.
        """
        super().__init__()
        if not isinstance(fn, nn.Module):
            raise TypeError("The `fn` argument must be an instance of `nn.Module`.")
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for PostNorm.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the module function and normalization.
        """
        x = self.fn(x)
        return self.norm(x)
