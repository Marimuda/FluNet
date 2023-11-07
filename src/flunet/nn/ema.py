import contextlib
from typing import Iterable, List

import torch
from torch import nn
from torch.nn import Parameter

DECAY_MINIMUM = 0.0
DECAY_MAXIMUM = 1.0
DECAY_ADJUSTMENT_BASE = 10


class LitEma(nn.Module):
    r"""A module to apply Exponential Moving Average (EMA) to model parameters. EMA helps in reducing the variance of the
    model predictions over time, leading to a more stable and generalizable model. This module tracks the EMA of a given
    model's parameters and allows for the EMA parameters to be used instead of the actual parameters when needed, such
    as during evaluation or inference.

    The module is initialized with a model whose parameters will be tracked.
    The `decay` determines the rate at which older observations are downweighted.
    If `use_num_updates` is true, then the decay rate is adjusted dynamically
    based on the number of updates.

    Args:
        parameters (Iterable[Parameter]): Parameters of the model to track.
        decay (float): Decay factor for EMA computation. Defaults to 0.9999.
        use_num_updates (bool): Whether to adjust decay dynamically based on the number
                                of updates. Defaults to True.

    Raises:
        ValueError: If the decay is not within the interval [0, 1].

    .. note::
        The EMA parameters are stored as buffers within the module to ensure
        that they are moved to the correct device alongside the model.

     Example:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> ema = LitEma(model.parameters())
        >>> for input, target in dataloader:
        >>>     optimizer.zero_grad()
        >>>     output = model(input)
        >>>     loss = nn.functional.mse_loss(output, target)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     ema.update()
        >>>
        >>>
        >>> with ema.average_parameters():
        >>>     validate(model, val_dataloader)
    """

    def __init__(self, parameters: Iterable[Parameter], decay: float = 0.9999, use_num_updates: bool = True) -> None:
        super().__init__()
        if not (DECAY_MINIMUM <= decay <= DECAY_MAXIMUM):
            raise ValueError(f"Decay must be between {DECAY_MINIMUM} and {DECAY_MAXIMUM}")

        self.decay = decay
        self.use_num_updates = use_num_updates
        self.num_updates = 0

        # Initialize shadow parameters
        self.shadow_params = [p.clone().detach() for p in parameters]
        self._params = list(parameters)

    def update(self):
        """Updates the shadow parameters with the current model parameters."""
        with torch.no_grad():
            decay = self._get_decay()

            for shadow_param, param in zip(self.shadow_params, self._params):
                shadow_param -= (1.0 - decay) * (shadow_param - param.data)

    def _get_decay(self) -> float:
        """Calculates the decay rate, possibly adjusting based on the number of updates."""
        if not self.use_num_updates:
            return self.decay
        self.num_updates += 1
        return min(self.decay, (1.0 + self.num_updates) / (DECAY_ADJUSTMENT_BASE + self.num_updates))

    def copy_to(self):
        """Copies the shadow parameters to the actual model parameters."""
        with torch.no_grad():
            for shadow_param, param in zip(self.shadow_params, self._params):
                param.data.copy_(shadow_param)

    def state_dict(self):
        """Returns a dictionary containing a whole state of the module."""
        state_dict = {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": [param.clone() for param in self.shadow_params],
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from `state_dict` into this module and its descendants."""
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]
        for shadow_param, param in zip(self.shadow_params, self._params):
            param.data.copy_(shadow_param)

    def store(self):
        """Stores the current parameters for later restoration."""
        self.saved_params = [param.clone() for param in self._params]

    def restore(self):
        """Restores the parameters stored with the `store` method."""
        with torch.no_grad():
            for saved_param, param in zip(self.saved_params, self._params):
                param.data.copy_(saved_param)

    @contextlib.contextmanager
    def average_parameters(self):
        """Context manager to temporarily replace model parameters with the EMA values."""
        self.store()
        self.copy_to()
        try:
            yield
        finally:
            self.restore()
