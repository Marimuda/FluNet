"""Custom class that redefines various types to increase clarity."""

from typing import Any, Callable, Dict, Iterable, List, Union

import torch
from numpy import ndarray
from torch import Tensor
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data
from torch_geometric.data.hetero_data import HeteroData

Key = Union[str, int]  # for dictionaries, we usually use strings or ints as keys
ConfigDict = Dict[Key, Any]  # A (potentially nested) dictionary containing the "params" section of the .yaml file
EntityDict = Dict[Key, Union[Dict, str]]  # potentially nested dictionary of entities
CheckpointDict = Dict[Key, Any]
ValueDict = Dict[Key, Any]
Result = Union[List, int, float, ndarray]
Shape = Union[int, Iterable, ndarray]

InputBatch = Union[Dict[Key, Tensor], Tensor, Batch, Data, HeteroData, None]
OutputTensorDict = Dict[Key, Tensor]
AggregationFunctionType = Callable[..., torch.Tensor]
