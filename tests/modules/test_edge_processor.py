from unittest.mock import Mock

import pytest
import torch
from torch import nn

from src.flunet.models import EdgeProcessorModule


class MockMLP(Mock):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def test_edgeprocessor_initialization():
    edgeprocessor = EdgeProcessorModule(
        in_dim_node=10, in_dim_edge=20, hidden_dim=15, hidden_layers=2, norm_type="BatchNorm1d", activation=nn.ReLU
    )
    assert isinstance(edgeprocessor, EdgeProcessorModule)


def test_edgeprocessor_forward(monkeypatch):
    # create a mock MLP object
    mock_mlp = MockMLP()

    mock_forward = Mock()

    # Specify the return value of the forward method
    mock_forward.return_value = torch.randn(50, 20)

    mock_mlp.forward = mock_forward

    # Use monkeypatch to replace the MLP class in the EdgeProcessorModule with the mock object
    monkeypatch.setattr("src.flunet.models.components.mlp.MLP", mock_mlp)

    edgeprocessor = EdgeProcessorModule(
        in_dim_node=10, in_dim_edge=20, hidden_dim=15, hidden_layers=2, norm_type="BatchNorm1d", activation=nn.ReLU
    )
    src = torch.randn(50, 10)
    dest = torch.randn(50, 10)
    edge_attr = torch.randn(50, 20)
    u = None
    batch = None
    output = edgeprocessor(src, dest, edge_attr, u, batch)
    assert output.size() == (50, 20)


def test_edgeprocessor_forward_with_invalid_input():
    edgeprocessor = EdgeProcessorModule(
        in_dim_node=10, in_dim_edge=20, hidden_dim=15, hidden_layers=2, norm_type="BatchNorm1d", activation=nn.ReLU
    )
    src = torch.randn(50, 5)
    dest = torch.randn(50, 5)
    edge_attr = torch.randn(50, 10)
    u = None
    batch = None
    with pytest.raises(RuntimeError):
        output = edgeprocessor(src, dest, edge_attr, u, batch)
