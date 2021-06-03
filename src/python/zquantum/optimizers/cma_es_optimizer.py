import numpy as np
from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    optimization_result,
    construct_history_info,
)
from scipy.optimize import OptimizeResult
import cma
from typing import Dict, Optional


class CMAESOptimizer(Optimizer):
    def __init__(self, sigma_0, options: Optional[Dict] = None):
        """
        Args:
            options: dictionary with options for the optimizer,
                please refer to https://github.com/CMA-ES/pycma documentation.

        """
        self.sigma_0 = sigma_0
        if options is None:
            options = {}
        self.options = options

    def minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: np.ndarray,
        keep_history: bool = True,
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

        if keep_history == False:
            raise ValueError("CMA-ES optimizer requires keep_history value to be True.")

        # Optimization Results Object
        cost_function = recorder(cost_function)

        strategy = cma.CMAEvolutionStrategy(initial_params, self.sigma_0, self.options)
        result = strategy.optimize(cost_function).result

        return optimization_result(
            opt_value=result.fbest,
            opt_params=result.xbest,
            nfev=result.evaluations,
            nit=result.iterations,
            cma_xfavorite=list(result.xfavorite),
            **construct_history_info(cost_function, keep_history)
        )
