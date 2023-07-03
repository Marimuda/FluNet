from torch_scatter import scatter_mean, scatter_min, scatter_max, scatter_sum, scatter_std


def get_aggregation_function(name: str) -> callable:
    if name == "mean":
        aggregation_function = scatter_mean
    elif name == "min":
        aggregation_function = lambda *args, **kwargs: scatter_min(*args, **kwargs)[0]
    elif name == "max":
        aggregation_function = lambda *args, **kwargs: scatter_max(*args, **kwargs)[0]
    elif name == "sum":
        aggregation_function = scatter_sum
    elif name == "std":
        aggregation_function = scatter_std
    else:
        raise ValueError(f"Unknown aggregation function '{name}'")
    return aggregation_function
