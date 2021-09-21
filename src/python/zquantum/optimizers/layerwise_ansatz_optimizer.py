import copy
from collections import defaultdict
from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.optimizer import (
    NestedOptimizer,
    Optimizer,
    extend_histories,
)
from zquantum.core.typing import RecorderFactory


def append_random_params(target_size: int, params: np.ndarray) -> np.ndarray:
    """
    Adds new random parameters to the `params` so that the size
    of the output is `target_size`.
    New parameters are sampled from a uniform distribution over [-pi, pi].

    Args:
        target_size: target number of parameters
        params: params that we want to extend
    """
    assert len(params) < target_size
    new_params = np.random.uniform(-np.pi, np.pi, target_size - len(params))
    return np.concatenate([params, new_params])


class LayerwiseAnsatzOptimizer(NestedOptimizer):
    @property
    def inner_optimizer(self) -> Optimizer:
        return self._inner_optimizer

    @property
    def recorder(self) -> RecorderFactory:
        return self._recorder

    def __init__(
        self,
        ansatz: Ansatz,
        inner_optimizer: Optimizer,
        min_layer: int,
        max_layer: int,
        n_layers_per_iteration: int = 1,
        parameters_initializer: Callable[
            [int, np.ndarray], np.ndarray
        ] = append_random_params,
        recorder: RecorderFactory = _recorder,
    ) -> None:
        """
        Args:
            inner_optimizer: optimizer used for optimization at each layer.
            min_layer: starting number of layers.
            max_layer: maximum number of layers, at which optimization should stop.
            TODO:

        Example usage (aka, what the heck are all these factories?):

            from functools import partial
            from zquantum.core.estimation import (
                estimate_expectation_values_by_averaging,
                allocate_shots_uniformly
            )
            from zquantum.core.cost_function import (
                substitution_based_estimation_tasks_factory,
                create_cost_function,
            )

            cost_hamiltonian = ...
            ansatz = ...

            estimation_preprocessors = [partial(allocate_shots_uniformly, number_of_shots=1000)]
            estimation_tasks_factory_generator = partial(
                substitution_based_estimation_tasks_factory,
                target_operator=cost_hamiltonian,
                estimation_preprocessors=estimation_preprocessors
            )
            cost_function_factory = partial(
                create_cost_function,
                backend=QuantumBackend,
                estimation_method=estimate_expectation_values_by_averaging,
                parameter_preprocessors=None,
            )

            initial_params = np.array([0.42, 4.2])
            inner_optimizer = ...

            optimizer = LayerwiseAnsatzOptimizer(
                ansatz,
                inner_optimizer,
                estimation_tasks_factory_generator,
                cost_function_factory,
                min_layer = ...,
                max_layer = ...,
            )

            opt_result = optimizer.minimize(initial_params)
        """

        assert 0 <= min_layer <= max_layer
        assert n_layers_per_iteration > 0
        self._ansatz = ansatz
        self._inner_optimizer = inner_optimizer
        self._min_layer = min_layer
        self._max_layer = max_layer
        self._n_layers_per_iteration = n_layers_per_iteration
        self._parameters_initializer = parameters_initializer
        self._recorder = recorder

    def _minimize(
        self,
        cost_function_factory: Callable[[Ansatz], CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all the parameters from the grid.

        Args:
            # TODO
            cost_function: object representing cost function we want to minimize
            inital_params: initial parameters for the cost function

        """
        ansatz = copy.deepcopy(self._ansatz)
        ansatz.number_of_layers = self._min_layer

        nit = 0
        nfev = 0
        histories = defaultdict(list)
        histories["history"] = []
        initial_params_per_iteration = initial_params
        for i in range(
            self._min_layer, self._max_layer + 1, self._n_layers_per_iteration
        ):
            assert ansatz.number_of_layers == i

            if i != self._min_layer:
                initial_params_per_iteration = self._parameters_initializer(
                    ansatz.number_of_params, optimal_params
                )

            cost_function = cost_function_factory(ansatz=ansatz)

            if keep_history:
                cost_function = self._recorder(cost_function)
            layer_results = self._inner_optimizer.minimize(
                cost_function, initial_params_per_iteration, keep_history=False
            )

            optimal_params: np.ndarray = layer_results.opt_params
            ansatz.number_of_layers += self._n_layers_per_iteration

            nfev += layer_results.nfev
            nit += layer_results.nit

            if keep_history:
                histories = extend_histories(cost_function, histories)

        del layer_results["history"]
        del layer_results["nit"]
        del layer_results["nfev"]

        return OptimizeResult(**layer_results, **histories, nfev=nfev, nit=nit)
