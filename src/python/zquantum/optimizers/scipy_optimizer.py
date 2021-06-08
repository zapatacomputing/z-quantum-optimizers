import numpy as np
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    optimization_result,
    construct_history_info,
)
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.typing import RecorderFactory
from typing import Optional, Tuple, Callable, Dict
import scipy


class ScipyOptimizer(Optimizer):
    def __init__(
        self,
        method: str,
        constraints: Optional[Tuple[Dict[str, Callable]]] = None,
        options: Optional[Dict] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            method: defines the optimization method
            constraints: list of constraints in the scipy compatible format.
            options: dictionary with additional options for the optimizer.
            recorder: recorder object which defines how to store the optimization history.

        """
        super().__init__(recorder=recorder)
        self.method = method
        if options is None:
            options = {}
        self.options = options
        self.constraints = [] if constraints is None else constraints

    def _minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: np.ndarray = None,
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

        jacobian = None
        if hasattr(cost_function, "gradient") and callable(
            getattr(cost_function, "gradient")
        ):
            jacobian = cost_function.gradient

        result = scipy.optimize.minimize(
            cost_function,
            initial_params,
            method=self.method,
            options=self.options,
            constraints=self.constraints,
            jac=jacobian,
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
