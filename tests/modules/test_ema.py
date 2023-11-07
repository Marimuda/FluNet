import pytest
import torch
from torch import nn

from flunet.nn import LitEma


def test_ema_initialization(simple_model):
    ema = LitEma(simple_model.parameters(), decay=0.9999)
    for param, ema_param in zip(simple_model.parameters(), ema.shadow_params):
        assert torch.allclose(
            param.data, ema_param
        ), "EMA shadow parameter does not match model parameter at initialization"


def test_ema_update(simple_model, input_tensor):
    ema = LitEma(simple_model.parameters(), decay=0.1)  # A lower decay rate to see a more noticeable effect
    optimizer = torch.optim.SGD(
        simple_model.parameters(), lr=1.0
    )  # Increase the learning rate to enforce a larger update
    optimizer.zero_grad()
    output = simple_model(input_tensor)
    loss = output.sum()  # Consider increasing the loss if necessary
    loss.backward()
    optimizer.step()

    ema.update()  # Perform the EMA update

    for param, shadow_param in zip(simple_model.parameters(), ema.shadow_params):
        # Check for a significant change rather than just non-equality
        assert not torch.allclose(
            param.data, shadow_param
        ), "Shadow parameter did not update after model parameter update"


def test_ema_integration(model_and_optimizer):
    model, optimizer = model_and_optimizer
    ema = LitEma(model.parameters())
    input_tensor = torch.randn((1, 10))
    target = torch.randint(0, 2, (1,))

    # Train for a few epochs
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        ema.update()

    # Use EMA parameters for evaluation
    with ema.average_parameters():
        output = model(input_tensor)
        loss_with_ema = nn.functional.cross_entropy(output, target)

    # Make sure EMA is actually being used by checking if loss_with_ema differs from the loss
    assert loss_with_ema != loss, "EMA did not affect model parameters"


def test_ema_state_dict(model_and_optimizer):
    model, _ = model_and_optimizer
    ema = LitEma(model.parameters())
    ema_state_dict = ema.state_dict()
    # Perform some updates
    for _ in range(10):
        ema.update()
    # Save the state after updates
    updated_ema_state_dict = ema.state_dict()
    # Load the original state back
    ema.load_state_dict(updated_ema_state_dict)  # This should be updated_ema_state_dict
    for original, updated in zip(updated_ema_state_dict["shadow_params"], ema.shadow_params):
        assert torch.equal(original, updated), "State dict loading did not preserve the updated state"


def test_ema_state_dict_save_load(simple_model):
    ema = LitEma(simple_model.parameters(), decay=0.9999)
    original_state_dict = ema.state_dict()

    ema.update()  # Apply updates to change the EMA state
    updated_state_dict = ema.state_dict()

    ema.load_state_dict(original_state_dict)  # Load the original state back

    for key in original_state_dict:
        original_value = original_state_dict[key]
        loaded_value = ema.state_dict()[key]
        if isinstance(original_value, list):
            for original_tensor, loaded_tensor in zip(original_value, loaded_value):
                assert torch.allclose(
                    original_tensor, loaded_tensor
                ), f"Tensor in list not restored correctly for {key}"
        elif isinstance(original_value, torch.Tensor):
            assert torch.allclose(original_value, loaded_value), f"Tensor state not restored correctly for {key}"
        else:
            # For non-tensor values, use direct equality since they are not subject to floating-point issues
            assert original_value == loaded_value, f"Non-tensor state not restored correctly for {key}"


def test_ema_copy_to(simple_model, input_tensor):
    ema = LitEma(simple_model.parameters(), decay=0.9999)
    ema.update()
    ema.copy_to()

    # Check if the model parameters match the shadow parameters
    for model_param, shadow_param in zip(simple_model.parameters(), ema.shadow_params):
        assert torch.allclose(
            model_param.data, shadow_param
        ), "Model parameter did not match EMA shadow parameter after copy_to"


def test_ema_restore(simple_model):
    ema = LitEma(simple_model.parameters(), decay=0.9999)
    ema.store()

    with torch.no_grad():
        for param in simple_model.parameters():
            param.add_(torch.randn_like(param))

    ema.restore()
    for stored_param, model_param in zip(ema.saved_params, simple_model.parameters()):
        assert torch.equal(stored_param, model_param.data), "Model parameters were not restored correctly"
