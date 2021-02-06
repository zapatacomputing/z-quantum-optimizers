from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import Optimizer, optimization_result
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.interfaces.functions import CallableStoringArtifacts
from scipy.optimize import OptimizeResult
from typing import Dict, Optional, List
import numpy as np
import random


class LayerwiseAnsatzOptimizer:
    def __init__(
        self,
        inner_optimizer: Optimizer,
        options: Optional[Dict] = None,
    ):
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
        self.inner_optimizer = inner_optimizer
        if "keep_value_history" not in self.options.keys():
            self.keep_value_history = False
        else:
            self.keep_value_history = self.options["keep_value_history"]
            del self.options["keep_value_history"]

    def minimize_lbl(
        self,
        cost_function: CallableStoringArtifacts,
        min_layer: int,
        max_layer: int,
        params_min_values: List[float],
        params_max_values: List[float],
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all the parameters from the grid.

        Args:
            cost_function(zquantum.core.interfaces.cost_function.CostFunction): object representing cost function we want to minimize
            inital_params (np.ndarray): initial parameters for the cost function

        Returns:
            OptimizeResults
        """

        if self.keep_value_history:
            cost_function = recorder(cost_function)

        cost_function.ansatz.number_of_layers = min_layer

        number_of_params = cost_function.ansatz.number_of_params
        initial_params = [
            params_min_values[i]
            + random.random() * (params_max_values[i] - params_min_values[i])
            for i in range(number_of_params)
        ]

        initial_params = np.array(initial_params)

        for _ in range(min_layer, max_layer + 1):
            layer_results = self.inner_optimizer.minimize(cost_function, initial_params)
            optimal_params = layer_results.opt_params
            new_layer_params = [
                params_min_values[i]
                + random.random() * (params_max_values[i] - params_min_values[i])
                for i in range(2)
            ]
            initial_params = np.array(optimal_params.tolist() + new_layer_params)
            cost_function.ansatz.number_of_layers += 1

        cost_function.ansatz.number_of_layers = max_layer

        return layer_results
