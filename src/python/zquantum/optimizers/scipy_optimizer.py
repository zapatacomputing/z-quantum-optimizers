from zquantum.core.interfaces.optimizer import Optimizer
from typing import List, Optional, Tuple, Callable, Dict
import scipy


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
        history = []

        def default_callback(params):
            history.append({"params": params})
            if self.keep_value_history:
                value = cost_function.evaluate(params)
                history[-1]["value"] = value
                print(f"Iteration {len(history)}: {value}", flush=True)
            else:
                print(f"iteration {len(history)}")
            print(f"{params}", flush=True)

        if callback is None:
            callback = default_callback
        cost_function_wrapper = lambda params: cost_function.evaluate(params).value
        result = scipy.optimize.minimize(
            cost_function_wrapper,
            initial_params,
            method=self.method,
            options=self.options,
            constraints=self.constraints,
            callback=callback,
            jac=cost_function.get_gradient,
        )

        result.opt_value = result.fun
        del result["fun"]
        result.opt_params = result.x
        del result["x"]
        result.history = history
        if "hess_inv" in result.keys():
            del result["hess_inv"]
        if "final_simplex" in result.keys():
            del result["final_simplex"]
        return result
