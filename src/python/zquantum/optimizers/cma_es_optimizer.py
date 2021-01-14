from copy import deepcopy

import numpy as np
from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import Optimizer, optimization_result
from scipy.optimize import OptimizeResult
import cma


class CMAESOptimizer(Optimizer):

    def __init__(self, options):
        """
        Args:
            options(dict): dictionary with options for the optimizer.

        Supported values for the options dictionary:
        Options:
            sigma_0(float): initial standard deviation. Required option
            keep_value_history(bool): boolean flag indicating whether the history of evaluations should be stored or not.
            **kwargs: other options, please refer to https://github.com/CMA-ES/pycma documentation.

        """
        options = deepcopy(options)
        if "sigma_0" not in options.keys():
            raise RuntimeError(
                'Error: CMAESOptimizer input options dictionary must contain "sigma_0" field'
            )
        else:
            self.sigma_0 = options.pop("sigma_0")
        self.options = options

        if "keep_value_history" in self.options.keys():
            del self.options["keep_value_history"]
            Warning(
                "CMA-ES always keeps track of the history, regardless of the keep_value_history flag."
            )

    def minimize(
        self, cost_function: CallableWithGradient, initial_params: np.ndarray
    ) -> OptimizeResult:
        """Minimize using the Covariance Matrix Adaptation Evolution Strategy
        (CMA-ES).

        Args:
            cost_function: object representing cost function we want to minimize
            initial_params: initial guess for the ansatz parameters.

        Returns:
            tuple: A tuple containing an optimization results dict and a numpy array
                with the optimized parameters.
        """

        # Optimization Results Object
        cost_function = recorder(cost_function)

        strategy = cma.CMAEvolutionStrategy(initial_params, self.sigma_0, self.options)
        result = strategy.optimize(cost_function).result

        return optimization_result(
            opt_value=result.fbest,
            opt_params=result.xbest,
            history=cost_function.history,
            nfev=result.evaluations,
            nit=result.iterations,
            cma_xfavorite=list(result.xfavorite),
        )
