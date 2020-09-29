from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import Optimizer, optimization_result
from zquantum.core.circuit import ParameterGrid
from scipy.optimize import OptimizeResult
from typing import Dict, Optional
import numpy as np


class GridSearchOptimizer(Optimizer):
    def __init__(self, grid: ParameterGrid, options: Optional[Dict] = None):
        """
        Args:
            grid(from zquantum.core.circuit.ParameterGrid): object defining for which parameters we want do the evaluations
            options(dict): dictionary with additional options for the optimizer.

        Supported values for the options dictionary:
        Options:
            keep_value_history(bool): boolean flag indicating whether the history of evaluations should be stored or not.
            
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
        self, cost_function: CallableWithGradient, initial_params: Optional[np.ndarray] = None
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all the parameters from the grid.

        Args:
            cost_function(zquantum.core.interfaces.cost_function.CostFunction): object representing cost function we want to minimize
            inital_params (np.ndarray): initial parameters for the cost function

        Returns:
            OptimizeResults
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
            opt_value=min_value, opt_params=optimal_params, nfev=nfev, nit=None, history=cost_function.history if self.keep_value_history else []
        )

    def get_values_grid(self, optimization_results: OptimizeResult) -> np.ndarray:
        """Shapes the values from the optimization results into the shape of the grid.

        Args:
            optimization_results (OptimizeResult): an optimization results dictionary
        
        Returns:
            numpy.ndarray: the values obtained at each grid point, shaped to have the same dimensions as the mesh grid
        """
        values = np.array([step["value"] for step in optimization_results["history"]])
        return np.reshape(values, self.grid.params_meshgrid[0].shape)
