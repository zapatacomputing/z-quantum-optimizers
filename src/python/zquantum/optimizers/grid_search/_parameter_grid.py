################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult
from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import Optimizer, optimization_result


class ParameterGrid:
    """A class representing a grid of parameter values to be used in a grid search.

    Args:
        param_ranges: ranges of the parameters describing the shape of the grid.
            Each range consist is of the form (min, max, step).

    Attributes:
        param_ranges (list): same as above.
    """

    def __init__(self, param_ranges: List[Tuple[float, float, float]]):
        self.param_ranges = param_ranges

    @property
    def params_list(self) -> List[np.ndarray]:
        grid_array = np.reshape(np.stack(self.params_meshgrid), (self.n_params, -1))

        return [grid_array[:, i].flatten() for i in range(grid_array.shape[1])]

    def to_dict(self) -> dict:
        return {"param_ranges": self.param_ranges}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["param_ranges"])

    @property
    def params_meshgrid(self) -> List[np.ndarray]:
        """
        Creates a meshgrid from the parameter ranges.
        """
        param_vectors: List[np.ndarray] = []

        for param_spec in self.param_ranges:
            param_vectors.append(np.arange(param_spec[0], param_spec[1], param_spec[2]))

        return np.meshgrid(*param_vectors, indexing="ij")

    @property
    def n_params(self) -> int:
        return len(self.param_ranges)


def build_uniform_param_grid(
    n_params_per_layer: int,
    n_layers: int = 1,
    min_value: float = 0.0,
    max_value: float = 2 * np.pi,
    step: float = np.pi / 5,
) -> ParameterGrid:
    """Builds a uniform grid of parameters.

    Args:
        n_params_per_layer: number of parameters for each layer
        n_layers: the number of layers to create parameters for
        min_value: the minimum value for the parameters
        max_value: the maximum value for the parameters
        step: the step size

    Returns:
        Points on a grid in parameter space.
    """

    n_params = n_params_per_layer * n_layers

    param_ranges = [(min_value, max_value, step)] * n_params
    return ParameterGrid(param_ranges)
