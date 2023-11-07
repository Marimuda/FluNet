import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def simple_model():
    """Provides a simple linear model from the PyTorch nn module.

    The fixture creates an instance of the Linear model with predefined input
    and output features size which is commonly used in tests that require a model.

    Returns:
        nn.Module: A PyTorch Linear model with 10 input features and 2 output features.
    """
    return nn.Linear(in_features=10, out_features=2)


@pytest.fixture(scope="module")
def input_tensor():
    """Provides a tensor of a specific shape compatible with the simple_model fixture.

    This fixture generates a single batch (size 1) of random data with 10 features,
    which can be used as input for the simple_model in tests.

    Returns:
        torch.Tensor: A random tensor of shape (1, 10).
    """
    return torch.randn((1, 10))


@pytest.fixture(scope="module")
def model_and_optimizer():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer
