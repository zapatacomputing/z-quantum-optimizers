import numpy as np
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    optimization_result,
    construct_history_info,
)
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.typing import RecorderFactory

from typing import Callable, Union
import scipy.optimize


class BasinHoppingOptimizer(Optimizer):
    def __init__(
        self,
        niter: int = 100,
        T: float = 1.0,
        stepsize: float = 0.5,
        minimizer_kwargs: Union[dict, None] = None,
        take_step: Union[Callable, None] = None,
        accept_test: Union[Callable, None] = None,
        interval: int = 50,
        disp: bool = False,
        niter_success: Union[int, None] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """The BasinHoppingOptimizer utilizes the scipy.optimize.basinhopping method
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html).
        It is intended to be used in conjunction with methods of
        scipy.optimize.minimize for local optimization.

        Args:
            niter: See scipy.optimize.basinhopping
            T: See scipy.optimize.basinhopping
            stepsize: See scipy.optimize.basinhopping
            minimizer_kwargs: See scipy.optimize.basinhopping
            take_step: See scipy.optimize.basinhopping
            accept_test: See scipy.optimize.basinhopping
            interval: See scipy.optimize.basinhopping
            disp: See scipy.optimize.basinhopping
            niter_success: See scipy.optimize.basinhopping
        """  # noqa
        super().__init__(recorder=recorder)
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.minimizer_kwargs = minimizer_kwargs
        self.take_step = take_step
        self.accept_test = accept_test
        self.interval = interval
        self.disp = disp
        self.niter_success = niter_success

    def _minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: np.ndarray = None,
        keep_history: bool = False,
    ):
        """
        Minimizes given cost function using functions from scipy.optimize.basinhopping.

        Args:
            cost_function(): python method which takes numpy.ndarray as input
            initial_params(np.ndarray): initial parameters to be used for optimization
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.
        """
        jacobian = None
        if hasattr(cost_function, "gradient") and callable(
            getattr(cost_function, "gradient")
        ):
            jacobian = cost_function.gradient
        if (
            self.minimizer_kwargs is not None
            and self.minimizer_kwargs.get("options", None) is not None
        ):
            self.minimizer_kwargs["options"]["jacobian"] = jacobian

        result = scipy.optimize.basinhopping(
            cost_function,
            initial_params,
            niter=self.niter,
            T=self.T,
            stepsize=self.stepsize,
            minimizer_kwargs=self.minimizer_kwargs,
            take_step=self.take_step,
            accept_test=self.accept_test,
            interval=self.interval,
            disp=self.disp,
            niter_success=self.niter_success,
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
