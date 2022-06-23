################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    construct_history_info,
    optimization_result,
)
from zquantum.core.typing import RecorderFactory


class ScipyOptimizer(Optimizer):
    def __init__(
        self,
        method: str,
        constraints: Optional[Tuple[Dict[str, Callable]]] = None,
        bounds: Union[
            scipy.optimize.Bounds,
            Sequence[Tuple[float, float]],
            None,
        ] = None,
        options: Optional[Dict] = None,
        callback: Optional[Callable] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Integration with scipy optimizers. Documentation for this module is minimal,
        please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Args:
            method: defines the optimization method
            constraints: list of constraints in the scipy compatible format.
            bounds: bounds for the parameters in the scipy compatible format.
            options: dictionary with additional options for the optimizer.
            callback: one-argument function (except for method="trust-constr")
                that takes in the parameter vector xk after every iteration.
            recorder: recorder object which defines how to store
                the optimization history.

        """  # noqa: E501
        super().__init__(recorder=recorder)
        self.method = method
        if options is None:
            options = {}
        self.options = options
        self.constraints = [] if constraints is None else constraints
        self.bounds = bounds
        self.callback = callback

    def _minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
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
        if isinstance(cost_function, CallableWithGradient) and callable(
            getattr(cost_function, "gradient")
        ):
            jacobian = cost_function.gradient

        result = scipy.optimize.minimize(
            cost_function,
            initial_params,
            method=self.method,
            options=self.options,
            constraints=self.constraints,
            bounds=self.bounds,
            jac=jacobian,
            callback=self.callback,
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
            **construct_history_info(cost_function, keep_history)  # type: ignore
        )
