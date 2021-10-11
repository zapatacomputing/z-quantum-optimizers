import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import skquant.interop.scipy as skq
from scipy.optimize import minimize
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    construct_history_info,
    optimization_result,
)
from zquantum.core.typing import RecorderFactory


class ScikitQuantOptimizer(Optimizer):
    def __init__(
        self,
        method: str,
        budget: Optional[int] = 10000,
        bounds: Optional[Union[List[int], List[List[int]], np.ndarray]] = None,
        recorder: RecorderFactory = _recorder,
    ) -> None:
        """
        Args:
            method: defines the optimization method
            bounds: list of constraints in the scipy compatible format.
            budget: maximum number of optimization iterations. Similar to SciPy's maxiter
            bounds: upper and lower bounds of the parameters. Each parameter has its own
                bounds. Tighter bounds lead to better optimization. When None is passed,
                (-1000, 1000) will be used as the bounds. If a single bound is provided, it
                will be used for all the passed parameters.
            recorder: recorder object which defines how to store the optimization history.

        """
        super().__init__(recorder=recorder)
        self.method = method
        self.budget = budget
        self.bounds = bounds

    def set_general_bounds(self, bounds: Union[List[int], np.ndarray]):
        np_cast_array = np.array(bounds)
        assert np_cast_array.shape == (1, 2)

        self.bounds = np_cast_array

    def _minimize(
        self,
        cost_function: Callable,
        initial_params: np.ndarray,
        keep_history: bool = False,
    ):
        """
        Minimizes given cost function using functions from skquant.opt.minimize.

        Args:
            cost_function: python method which takes numpy.ndarray as input
            initial_params: initial parameters to be used for optimization
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """
        if self.bounds is None:
            warnings.warn(
                "Providing bounds for parameters is HIGHLY recommended! "
                + "(-1000, 1000) will be used as the bounds, which can hinder convergence.",
            )

            self.bounds = np.array([-1000, 1000])

        if len(self.bounds) == 1:
            self.bounds = np.tile(
                np.array(self.bounds, dtype=float), (initial_params.shape[0], 1)
            )
        elif len(self.bounds) != len(initial_params):
            raise ValueError(
                "Insufficient number of bounds provided. Need {} but found {}.".format(
                    len(initial_params), len(self.bounds)
                )
            )

        result = minimize(
            fun=cost_function,
            x0=initial_params,
            method=getattr(skq, self.method.lower()),
            bounds=self.bounds,
            options={"budget": self.budget},
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
