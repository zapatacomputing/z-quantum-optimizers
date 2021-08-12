import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import skquant
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    construct_history_info,
    optimization_result,
)
from zquantum.core.typing import RecorderFactory


class ScikitQuantOptimizers(Optimizer):
    def __init__(
        self,
        method: str,
        budget: Optional[int] = 10000,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            method: defines the optimization method
            bounds: list of constraints in the scipy compatible format.
            budget: maximum number of optimization iterations. Similar to SciPy's maxiter
            recorder: recorder object which defines how to store the optimization history.

        """
        super().__init__(recorder=recorder)
        self.method = method
        self.budget = budget

    def _minimize(
        self,
        cost_function: Callable,
        initial_params: np.ndarray,
        bounds: Optional[Union[List[List[int]], np.ndarray]] = None,
        keep_history: bool = False,
    ):
        """
        Minimizes given cost function using functions from scipy.minimize.

        Args:
            cost_function: python method which takes numpy.ndarray as input
            initial_params: initial parameters to be used for optimization
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """

        if bounds is None:
            warnings.warn(
                "Providing bounds for parameters is HIGHLY recommended! "
                + "The optimization will take place in an unconstrained manner.",
            )

            bounds = np.tile(np.array([-np.inf, np.inf]), [initial_params.shape[0], 1])

        result, _ = skquant.opt.minimize(
            cost_function, initial_params, bounds, self.budget, method=self.method
        )

        opt_value = result.fun
        opt_params = result.x

        nit = result.get("nit", None)
        nfev = result.get("nfev", None)

        return optimization_result(
            opt_value=opt_value,
            opt_params=opt_params,
            nit=nit,
            nfev=nfev,
            **construct_history_info(cost_function, keep_history)
        )
