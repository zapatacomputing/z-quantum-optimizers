from zquantum.core.interfaces.optimizer import Optimizer
from scipy.optimize import OptimizeResult
import cma
import copy

class CMAESOptimizer(Optimizer):

    def __init__(self, options):
        if "sigma_0" not in options.keys():
            raise RuntimeError("Error: CMAESOptimizer input options dictionary must contain \"sigma_0\" field")
        else:
            self.sigma_0 = options.pop('sigma_0')
        self.options = options

        if "keep_value_history" in self.options.keys():
            del self.options["keep_value_history"]
            Warning("CMA-ES always keeps track of the history, regardless of the keep_value_history flag.")

    def minimize(self, cost_function, initial_params):
        """Minimize using the Covariance Matrix Adaptation Evolution Strategy
        (CMA-ES).

        Args:
            cost_function(zquantum.core.interfaces.cost_function.CostFunction): object representing cost function we want to minimize
            initial_params (numpy.ndarray): initial guess for the ansatz parameters.

        Returns:
            tuple: A tuple containing an optimization results dict and a numpy array
                with the optimized parameters.
        """

        # Optimization Results Object
        history = []

        def wrapped_cost_function(params):
            value = cost_function.evaluate(params)
            history.append(cost_function.evaluations_history[-1])
            print(f'Iteration {len(history)}: {value}', flush=True)
            print(f'{params}', flush=True)
            return value

        strategy = cma.CMAEvolutionStrategy(initial_params, self.sigma_0, self.options)
        result = strategy.optimize(wrapped_cost_function).result

        optimization_results = {}
        optimization_results['opt_value'] = result.fbest
        optimization_results['opt_params'] = result.xbest
        optimization_results['history'] = history
        optimization_results['nfev'] = result.evaluations
        optimization_results['nit'] = result.iterations
        optimization_results['cma_xfavorite'] = list(result.xfavorite)

        return OptimizeResult(optimization_results)
