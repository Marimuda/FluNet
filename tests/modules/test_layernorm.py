import pytest
import torch
from torch import nn

from flunet.nn.norm import (  # Replace 'your_module' with the actual name of your module
    LayerNorm,
    PostNorm,
    PreNorm,
)

# Constants for tests
DIM = 10
BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = 2, DIM, 5, 5


@pytest.fixture
def input_tensor_4d():
    # Arrange
    return torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture
def mock_conv_module():
    # Arrange
    return nn.Conv2d(DIM, DIM, kernel_size=3, padding=1)


def test_layer_norm_should_maintain_input_shape(input_tensor_4d):
    # Arrange
    layer_norm = LayerNorm(DIM)

    # Act
    normalized_tensor = layer_norm(input_tensor_4d)

    # Assert
    assert normalized_tensor.shape == input_tensor_4d.shape


def test_layer_norm_should_initialize_parameters_correctly():
    # Arrange
    layer_norm = LayerNorm(DIM)

    # Act and Assert
    assert torch.allclose(layer_norm.g, torch.ones(1, DIM, 1, 1))
    assert torch.allclose(layer_norm.b, torch.zeros(1, DIM, 1, 1))


def test_prenorm_should_apply_module_function_after_normalization(input_tensor_4d, mock_conv_module):
    # Arrange
    prenorm = PreNorm(DIM, mock_conv_module)

    # Act
    output = prenorm(input_tensor_4d)

    # Assert
    assert output.shape == (BATCH_SIZE, DIM, HEIGHT, WIDTH)


def test_postnorm_should_apply_module_function_before_normalization(input_tensor_4d, mock_conv_module):
    # Arrange
    postnorm = PostNorm(DIM, mock_conv_module)

    # Act
    output = postnorm(input_tensor_4d)

    # Assert
    assert output.shape == (BATCH_SIZE, DIM, HEIGHT, WIDTH)


def test_layer_norm_should_raise_value_error_on_invalid_input_shape():
    # Arrange
    layer_norm = LayerNorm(DIM)
    invalid_input = torch.randn(2, DIM)  # Invalid shape, missing two dimensions

    # Act and Assert
    with pytest.raises(ValueError):
        layer_norm(invalid_input)


def test_layer_norm_should_compute_correct_normalization(input_tensor_4d):
    # Arrange
    layer_norm = LayerNorm(DIM)

    # Act
    output = layer_norm(input_tensor_4d)
    var = torch.var(input_tensor_4d, dim=1, keepdim=True, unbiased=False)
    mean = torch.mean(input_tensor_4d, dim=1, keepdim=True)
    expected_output = (input_tensor_4d - mean) / torch.sqrt(var + layer_norm.eps)

    # Assert
    assert torch.allclose(output, expected_output * layer_norm.g + layer_norm.b)


def test_layer_norm_parameters_should_update_during_training(input_tensor_4d, mock_conv_module):
    # Arrange
    layer_norm = LayerNorm(DIM)
    prenorm = PreNorm(DIM, mock_conv_module)
    optimizer = torch.optim.SGD(prenorm.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    original_g = layer_norm.g.clone()
    original_b = layer_norm.b.clone()

    # Act
    output = prenorm(input_tensor_4d)
    loss = loss_fn(output, torch.randn_like(output))
    loss.backward()
    optimizer.step()

    # Assert
    updated_g = next(prenorm.norm.parameters())
    updated_b = next(iter(prenorm.norm.parameters()))

    assert not torch.allclose(updated_g, original_g)
    assert not torch.allclose(updated_b, original_b)


def test_layer_norm_should_handle_zero_variance_case(input_tensor_4d):
    # Arrange
    layer_norm = LayerNorm(DIM)
    input_tensor_4d.fill_(1.0)  # This will have zero variance

    # Act
    output = layer_norm(input_tensor_4d)

    # Assert
    # Since the variance is zero, the output should be equal to the bias term.
    assert torch.allclose(output, layer_norm.b)
