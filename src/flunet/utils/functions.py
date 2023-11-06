import torch
from torch_scatter import (
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_std,
    scatter_sum,
)

from flunet.utils.types import AggregationFunctionType


def get_aggregation_function(name: str) -> AggregationFunctionType:
    """Retrieves a scatter aggregation function based on the provided name.

    Args:
        name (str): The name of the aggregation function. Possible values are
                    "mean", "min", "max", "sum", and "std".

    Returns:
        AggregationFunctionType: A callable aggregation function that matches
                                 the name provided.

    Raises:
        ValueError: If the given name does not match any known aggregation
                    functions.
    """

    def min_wrapper(*args, **kwargs) -> torch.Tensor:
        """Wrapper for scatter_min that returns only the minimum values tensor."""
        return scatter_min(*args, **kwargs)[0]

    def max_wrapper(*args, **kwargs) -> torch.Tensor:
        """Wrapper for scatter_max that returns only the maximum values tensor."""
        return scatter_max(*args, **kwargs)[0]

    match name:
        case "mean":
            return scatter_mean
        case "min":
            return min_wrapper
        case "max":
            return max_wrapper
        case "sum":
            return scatter_sum
        case "std":
            return scatter_std
        case _:
            raise ValueError(f"Unknown aggregation function '{name}'")
