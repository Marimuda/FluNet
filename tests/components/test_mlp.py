import pytest
import torch
from src.flunet.models.components.mlp import MLP

def test_mlp_initialization():
    mlp = MLP(in_dim=10, out_dim=20, hidden_dim=15, hidden_layers=2, norm_type='BatchNorm1d')
    assert mlp.model[0].in_features == 10
    assert mlp.out_dim == 20
    assert mlp.hidden_dim == 15
    assert mlp.hidden_layers == 2
    assert mlp.norm_type == 'BatchNorm1d'

def test_mlp_forward():
    mlp = MLP(in_dim=10, out_dim=20, hidden_dim=15, hidden_layers=2, norm_type='BatchNorm1d')
    input_data = torch.randn(50, 10)
    output = mlp(input_data)
    assert output.size() == (50, 20)

def test_mlp_forward_with_invalid_input():
    mlp = MLP(in_dim=10, out_dim=20, hidden_dim=15, hidden_layers=2, norm_type='BatchNorm1d')
    input_data = torch.randn(50, 5)
    with pytest.raises(RuntimeError):
        output = mlp(input_data)
