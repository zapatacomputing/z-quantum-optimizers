from zquantum.core.interfaces.optimizer import Optimizer
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.circuit import ParameterGrid
from scipy.optimize import OptimizeResult
from typing import Dict, Optional, Callable
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
        self, cost_function: CostFunction, initial_params: Optional[np.ndarray] = None
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

        for params in self.grid.params_list:
            value = cost_function.evaluate(params)
            if self.keep_value_history:
                history.append({"params": params, "value": value})
            nfev += 1
            if min_value is None or value < min_value:
                min_value = value
                optimal_params = params

        optimization_results = {}
        optimization_results["opt_value"] = min_value
        optimization_results["opt_params"] = optimal_params
        optimization_results["history"] = history
        optimization_results["nfev"] = nfev
        optimization_results["nit"] = None

        return OptimizeResult(optimization_results)

    def get_values_grid(self, optimization_results: OptimizeResult) -> np.ndarray:
        """Shapes the values from the optimization results into the shape of the grid.

        Args:
            optimization_results (OptimizeResult): an optimization results dictionary
        
        Returns:
            numpy.ndarray: the values obtained at each grid point, shaped to have the same dimensions as the mesh grid
        """
        values = np.array([step["value"] for step in optimization_results["history"]])
        return np.reshape(values, self.grid.params_meshgrid[0].shape)
