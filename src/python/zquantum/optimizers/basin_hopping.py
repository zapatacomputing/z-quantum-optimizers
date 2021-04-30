from zquantum.core.interfaces.optimizer import Optimizer, optimization_result
from zquantum.core.history.recorder import recorder
from typing import Callable, Dict, Union
import scipy.optimize


class BasinHoppingOptimizer(Optimizer):
    def __init__(
        self,
        keep_value_history: bool = False,
        niter: int = 100,
        T: float = 1.0,
        stepsize: float = 0.5,
        minimizer_kwargs: Union[dict, None] = None,
        take_step: Union[Callable, None] = None,
        accept_test: Union[Callable, None] = None,
        interval: int = 50,
        disp: bool = False,
        niter_success: Union[int, None] = None,
    ):
        """The BasinHoppingOptimizer utilizes the scipy.optimize.basinhopping method
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html). It is intended
        to be used in conjunction with methods of scipy.optimize.minimize for local optimization.

        Args:
            keep_value_history: determines whether or not the cost function records when it is evaluated
            niter: See scipy.optimize.basinhopping
            T: See scipy.optimize.basinhopping
            stepsize: See scipy.optimize.basinhopping
            minimizer_kwargs: See scipy.optimize.basinhopping
            take_step: See scipy.optimize.basinhopping
            accept_test: See scipy.optimize.basinhopping
            interval: See scipy.optimize.basinhopping
            disp: See scipy.optimize.basinhopping
            niter_success: See scipy.optimize.basinhopping
        """
        self.keep_value_history = keep_value_history
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.minimizer_kwargs = minimizer_kwargs
        self.take_step = take_step
        self.accept_test = accept_test
        self.interval = interval
        self.disp = disp
        self.niter_success = niter_success

    def minimize(self, cost_function, initial_params=None, callback=None):
        """
        Minimizes given cost function using functions from scipy.optimize.basinhopping.

        Args:
            cost_function(): python method which takes numpy.ndarray as input
            initial_params(np.ndarray): initial parameters to be used for optimization
            callback(): callback function. If none is provided, a default one will be used.

        Returns:
            optimization_results(scipy.optimize.OptimizeResults): results of the optimization.
        """

        if self.keep_value_history:
            cost_function = recorder(cost_function)

        jacobian = None
        if hasattr(cost_function, "gradient") and callable(
            getattr(cost_function, "gradient")
        ):
            jacobian = cost_function.gradient
        if self.minimizer_kwargs is not None:
            if self.minimizer_kwargs.get("options", None) is not None:
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
            history=cost_function.history if self.keep_value_history else [],
        )
