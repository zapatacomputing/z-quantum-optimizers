import numpy as np
from zquantum.core.history.recorder import recorder as _recorder
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
    def __init__(
        self, sigma_0: float, options: Optional[Dict] = None, recorder=_recorder
    ):
        """
        Integration with CMA-ES optimizer: https://github.com/CMA-ES/pycma .
        Args:
            sigma_0: please refer to https://github.com/CMA-ES/pycma documentation.
            options: dictionary with options for the optimizer,
                please refer to https://github.com/CMA-ES/pycma documentation.
            recorder: recorder object which defines how to store the optimization history.
        """
        super().__init__(recorder=recorder)
        self.sigma_0 = sigma_0
        if options is None:
            options = {}
        self.options = options

    def _minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """Minimize using the Covariance Matrix Adaptation Evolution Strategy
        (CMA-ES).

        Note:
            Original CMA-ES implementation stores optimization history by default.
            This is a separate mechanism from the one controlled by recorder, and
            therefore is turned on even if keep_history is set to false, which might
            lead to memory issues in some extreme cases.
            However, we expose only the recording performed using provided recorder.

        Args:
            cost_function: object representing cost function we want to minimize
            initial_params: initial guess for the ansatz parameters.
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.
        """
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
