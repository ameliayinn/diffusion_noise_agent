# tests/test_unet.py
import pytest
import torch
from unet import UNetSimulationWithMoE

@pytest.fixture
def model():
    return UNetSimulationWithMoE(time_emb_dim=128, image_size=8)

def test_valid_input(model):
    input = torch.randn(2, 1, 8, 8)
    output = model(input, torch.tensor([1,1]))
    assert output.shape == (2, 1, 8, 8)

def test_invalid_input(model):
    with pytest.raises(ValueError):
        model(torch.randn(2, 1), torch.tensor([1,1]))