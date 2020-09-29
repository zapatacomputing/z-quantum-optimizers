from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import Optimizer, optimization_result
from qiskit.aqua.components.optimizers import SPSA, ADAM
from scipy.optimize import OptimizeResult


class QiskitOptimizer(Optimizer):
    def __init__(self, method, options={}):
        """
        Args:
            method(str): specifies optimizer to be used. Currently supports "ADAM", "AMSGRAD" and "SPSA".
            options(dict): dictionary with additional options for the optimizer.

        Supported values for the options dictionary:
        Options:
            keep_value_history(bool): boolean flag indicating whether the history of evaluations should be stored or not.
            **kwargs: options specific for particular scipy optimizers.
            
        """

        self.method = method
        self.options = options
        if "keep_value_history" not in self.options.keys():
            self.keep_value_history = False
        else:
            self.keep_value_history = self.options["keep_value_history"]
            del self.options["keep_value_history"]
            Warning(
                "Orquestra does not support keeping history of the evaluations yet."
            )

    def minimize(self, cost_function: CallableWithGradient, initial_params=None):
        """
        Minimizes given cost function using optimizers from Qiskit Aqua.

        Args:
            cost_function(): python method which takes numpy.ndarray as input
            initial_params(np.ndarray): initial parameters to be used for optimization
        
        Returns:
            optimization_results(scipy.optimize.OptimizeResults): results of the optimization.
        """

        if self.method == "SPSA":
            optimizer = SPSA(**self.options)
        elif self.method == "ADAM" or self.method == "AMSGRAD":
            if self.method == "AMSGRAD":
                self.options["amsgrad"] = True
            optimizer = ADAM(**self.options)

        statistics = {"call_count": 0}

        def _cost_function_wrapper(params):
            statistics["call_count"] += 1
            return cost_function(params)

        number_of_variables = len(initial_params)

        solution, value, nit = optimizer.optimize(
            num_vars=number_of_variables,
            objective_function=_cost_function_wrapper,
            initial_point=initial_params,
            gradient_function=cost_function.gradient,
        )

        return optimization_result(
            opt_value=value, opt_params=solution, history=[], nfev=statistics["call_count"], nit=nit
        )
