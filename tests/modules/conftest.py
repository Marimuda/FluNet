import pytest
import torch
from torch import nn

from flunet.nn.norm import LayerNorm


@pytest.fixture(scope="module")
def input_tensor_4d():
    """Provides a 4-dimensional tensor simulating a batch of multi-channel images, which can be used as input for
    convolutional layers during tests.

    The tensor shape follows the convention (batch_size, channels, height, width),
    commonly used for image data in convolutional neural networks.

    This fixture generates a batch of 10 tensors, each with 5 channels and
    spatial dimensions of 10x10, filled with random data.

    Returns:
        torch.Tensor: A random 4D tensor of shape (10, 5, 10, 10).
    """
    return torch.randn(10, 5, 10, 10)


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
def simple_model():
    """Provides a simple linear model from the PyTorch nn module.

    The fixture creates an instance of the Linear model with predefined input
    and output features size which is commonly used in tests that require a model.

    Returns:
        nn.Module: A PyTorch Linear model with 10 input features and 2 output features.
    """
    return nn.Linear(in_features=10, out_features=2)


@pytest.fixture(scope="module")
def model_and_optimizer(simple_model):
    """Provides a simple linear model along with an SGD optimizer, suitable for testing.

    This fixture uses the `simple_model` fixture to create a consistent linear model
    across tests within the same module. It initializes a Stochastic Gradient Descent (SGD)
    optimizer with a learning rate of 0.1, which can be used to perform optimization
    steps during testing.

    The 'module' scope ensures this fixture is only executed once per test module,
    which means all tests in the module will use the same model and optimizer instances.

    Args:
        simple_model (nn.Module): A PyTorch Linear model fixture with predefined input/output features.

    Returns:
        tuple: A tuple containing:
            - nn.Module: A PyTorch Linear model provided by the `simple_model` fixture.
            - torch.optim.Optimizer: An SGD optimizer configured for the provided model.
    """
    optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.1)
    return simple_model, optimizer


@pytest.fixture
def layer_norm_module():
    normalized_shape = (16,)
    layer_norm = LayerNorm(normalized_shape)
    return layer_norm
