import torch
from torch import nn


class Normalizer(nn.Module):
    r"""A module for normalizing input data and accumulating statistics.

    The module normalizes input data based on accumulated statistics and
    provides an option to update these statistics with each forward pass.

    Args:
        size (int): The size of the data to be normalized.
        name (str): A name for the normalizer instance.
        max_accumulation (float, optional): The maximum value for accumulation.
            Default is \(10^6\).
        std_epsilon (float, optional): A small value to ensure numerical
            stability while normalizing. Default is \(1e-8\).

    Shapes:
        - **input:** batched_data: \((B, L, N)\), where \(B\) is the batch size,
          \(L\) is the length of the trajectory, and \(N\) is the number of features.
        - **output:** Same as input.
    """

    def __init__(self, size: int, name: str, max_accumulation: float = 10**6, std_epsilon: float = 1e-8):
        super().__init__()
        self.name = name

        self.register_buffer("max_accumulation", torch.tensor(max_accumulation))
        self.register_buffer("std_epsilon", torch.tensor(std_epsilon))

        self.register_buffer("acc_count", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("num_acc", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("acc_sum", torch.zeros(size, dtype=torch.float32))
        self.register_buffer("acc_sum_squared", torch.zeros(size, dtype=torch.float32))

    def forward(self, batched_data: torch.Tensor, accumulate: bool = True) -> torch.Tensor:
        """Normalizes input data and accumulates statistics."""
        if accumulate and self.num_acc < self.max_accumulation:
            self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_eps()

    def inverse(self, normalized_batch_data: torch.Tensor) -> torch.Tensor:
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_eps() + self._mean()

    def _accumulate(self, batched_data: torch.Tensor):
        batch_size, length_traj, _ = batched_data.shape
        batched_data = batched_data.reshape(batch_size * length_traj, -1)

        data_sum = torch.sum(batched_data, dim=0)
        data_sum_squared = torch.sum(batched_data**2, dim=0)
        self.acc_sum += data_sum
        self.acc_sum_squared += data_sum_squared
        self.acc_count += torch.tensor(batch_size * length_traj).to(self.acc_count.device)
        self.num_acc += torch.tensor(1.0).to(self.num_acc.device)

    def _mean(self):
        safe_count = torch.maximum(self.acc_count, torch.tensor(1.0).to(self.acc_count.device))
        return self.acc_sum / safe_count

    def _std_with_eps(self):
        safe_count = torch.maximum(self.acc_count, torch.ones_like(self.acc_count))
        std = torch.sqrt(self.acc_sum_squared / safe_count - self._mean() ** 2)
        return torch.maximum(std, self.std_epsilon)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
