from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    optimization_result,
    construct_history_info,
)
from ._parameter_grid import ParameterGrid
from scipy.optimize import OptimizeResult
from typing import Dict, Optional
import numpy as np


class GridSearchOptimizer(Optimizer):
    def __init__(self, grid: ParameterGrid):
        """
        Args:
            grid: object defining for which parameters we want do the evaluations
        """
        self.grid = grid

    def minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: Optional[np.ndarray] = None,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all the parameters from the grid.

        Args:
            cost_function: object representing cost function we want to minimize
            inital_params: initial parameters for the cost function
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """
        if initial_params is not None and len(initial_params) != 0:
            Warning("Grid search doesn't use initial parameters, they will be ignored.")

        min_value = None
        nfev = 0

        if keep_history:
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
            **construct_history_info(cost_function, keep_history)
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
