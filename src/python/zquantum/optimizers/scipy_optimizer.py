from zquantum.core.interfaces.optimizer import Optimizer, optimization_result
from zquantum.core.history.recorder import recorder
from typing import Optional, Tuple, Callable, Dict
import scipy
import scipy.optimize


class ScipyOptimizer(Optimizer):
    def __init__(
        self,
        method: str,
        constraints: Optional[Tuple[Dict[str, Callable]]] = None,
        options=None,
    ):
        """
        Args:
            method(from zquantum.core.circuit.ParameterGrid): object defining for which parameters we want do the evaluations
            constraints(Tuple[Dict[str, Callable]]): List of constraints in the scipy format.
            options(dict): dictionary with additional options for the optimizer.

        Supported values for the options dictionary:
        Options:
            keep_value_history(bool): boolean flag indicating whether the history of evaluations should be stored or not.
            **kwargs: options specific for particular scipy optimizers.

        """
        self.method = method
        if options is None:
            options = {}
        self.options = options
        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints
        if "keep_value_history" not in self.options.keys():
            self.keep_value_history = False
        else:
            self.keep_value_history = self.options["keep_value_history"]
            del self.options["keep_value_history"]

    def minimize(self, cost_function, initial_params=None, callback=None):
        """
        Minimizes given cost function using functions from scipy.minimize.

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
            history=cost_function.history if self.keep_value_history else [],
        )
