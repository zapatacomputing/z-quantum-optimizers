from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import Optimizer, optimization_result
from scipy.optimize import OptimizeResult
from typing import Dict, Optional, List, Tuple
import numpy as np


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
    def params_meshgrid(self) -> Tuple[np.ndarray, ...]:
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


class GridSearchOptimizer(Optimizer):
    def __init__(self, grid: ParameterGrid, options: Optional[Dict] = None):
        """
        Args:
            grid: object defining for which parameters we want do the evaluations
            options: dictionary with additional options for the optimizer.

        Supported values for the options dictionary:
        Options:
            keep_value_history: boolean flag indicating whether the history of evaluations should be stored or not.

        """
        if options is None:
            options = {}
        self.options = options
        self.grid = grid
        if "keep_value_history" not in self.options.keys():
            self.keep_value_history = False
        else:
            self.keep_value_history = self.options["keep_value_history"]
            del self.options["keep_value_history"]

    def minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: Optional[np.ndarray] = None,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all the parameters from the grid.

        Args:
            cost_function: object representing cost function we want to minimize
            inital_params: initial parameters for the cost function
        """
        if initial_params is not None and len(initial_params) != 0:
            Warning("Grid search doesn't use initial parameters, they will be ignored.")
        history = []
        min_value = None
        nfev = 0

        if self.keep_value_history:
            cost_function = recorder(cost_function)

        for params in self.grid.params_list:
            value = cost_function(params)
            nfev += 1
            if min_value is None or value < min_value:
                min_value = value
                optimal_params = params

        return optimization_result(
            opt_value=min_value,
            opt_params=optimal_params,
            nfev=nfev,
            nit=None,
            history=cost_function.history if self.keep_value_history else [],
        )

    def get_values_grid(self, optimization_results: OptimizeResult) -> np.ndarray:
        """Shapes the values from the optimization results into the shape of the grid.

        Args:
            optimization_results: an optimization results dictionary

        Returns:
            numpy.ndarray: the values obtained at each grid point, shaped to have the same dimensions as the mesh grid
        """
        values = np.array([step["value"] for step in optimization_results["history"]])
        return np.reshape(values, self.grid.params_meshgrid[0].shape)
