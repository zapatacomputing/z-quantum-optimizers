from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import Optimizer, construct_history_info
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.cost_function import (
    CostFunction,
    EstimationTasksFactory,
    ParameterPreprocessor,
)
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    construct_history_info,
    MetaOptimizer,
)
from zquantum.core.typing import RecorderFactory
from scipy.optimize import OptimizeResult
from typing import Optional, Union, Callable
import numpy as np
from functools import partial
import copy
from collections import defaultdict
from typing_extensions import Protocol
import abc
from zquantum.core.utils import ValueEstimate


class LayerwiseAnsatzOptimizer:
    def __init__(
        self,
        inner_optimizer: Optimizer,
        min_layer: int,
        max_layer: int,
        n_layers_per_iteration: int = 1,
        parameters_initializer: Optional[Callable] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            inner_optimizer: optimizer used for optimization at each layer.
            min_layer: starting number of layers.
            max_layer: maximum number of layers, at which optimization should stop.
            recorder: recorder object which defines how to store the optimization history.
        """
        assert 0 <= min_layer <= max_layer
        assert n_layers_per_iteration > 0
        self.inner_optimizer = inner_optimizer
        self.min_layer = min_layer
        self.max_layer = max_layer
        self.n_layers_per_iteration = n_layers_per_iteration
        self.parameters_initializer: Callable
        if parameters_initializer is None:
            self.parameters_initializer = partial(np.random.uniform, -np.pi, np.pi)
        else:
            self.parameters_initializer = parameters_initializer
        self.recorder = recorder

    def minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all the parameters from the grid.

        Note:
            Cost function needs to have `ansatz` property

        Args:
            cost_function(zquantum.core.interfaces.cost_function.CostFunction): object representing cost function we want to minimize
            inital_params (np.ndarray): initial parameters for the cost function

        """
        # Since this optimizer modifies the ansatz which is a part of the input cost function
        # we use copy of it instead.
        cost_function = copy.deepcopy(cost_function)

        if not hasattr(cost_function, "ansatz"):
            raise ValueError("Provided cost function needs to have ansatz property.")
        if keep_history:
            cost_function = self.recorder(cost_function)

        cost_function.ansatz.number_of_layers = self.min_layer

        number_of_params = cost_function.ansatz.number_of_params
        if initial_params is None:
            initial_params = self.parameters_initializer(number_of_params)

        for i in range(self.min_layer, self.max_layer + 1, self.n_layers_per_iteration):
            # keep_history is set to False, as the cost function is already being recorded
            # if keep_history is specified.
            if i != self.min_layer:
                new_layer_params = self.parameters_initializer(
                    cost_function.ansatz.number_of_params - len(optimal_params)
                )
                initial_params = np.concatenate([optimal_params, new_layer_params])

            layer_results = self.inner_optimizer.minimize(
                cost_function, initial_params, keep_history=False
            )
            optimal_params: np.ndarray = layer_results.opt_params
            cost_function.ansatz.number_of_layers += self.n_layers_per_iteration

        # layer_results["history"] will be empty as inner_optimizer was used with
        # keep_history false.
        del layer_results["history"]

        return OptimizeResult(
            **layer_results, **construct_history_info(cost_function, keep_history)
        )


class _EstimationTasksFactoryCreatorWithAnsatz(Protocol):
    @abc.abstractmethod
    def __call__(self, ansatz: Ansatz) -> EstimationTasksFactory:
        """Produce estimation task factories for given ansatz."""


class _CostFunctionFactory(Protocol):
    @abc.abstractmethod
    def __call__(
        self, estimation_tasks_factory: EstimationTasksFactory
    ) -> CostFunction:
        """Creates a cost function from an EstimationTasksFactory object."""


def _get_new_layer_params(number_of_params: int, old_params: np.ndarray) -> np.ndarray:
    new_layer_params = np.random.uniform(-np.pi, np.pi, number_of_params)
    return np.concatenate([old_params, new_layer_params])


class LayerwiseAnsatzOptimizerWithFactories(MetaOptimizer):
    def __init__(
        self,
        ansatz: Ansatz,
        inner_optimizer: Optimizer,
        estimation_tasks_factory: _EstimationTasksFactoryCreatorWithAnsatz,
        cost_function_factory: _CostFunctionFactory,
        min_layer: int,
        max_layer: int,
        n_layers_per_iteration: int = 1,
        parameters_initializer: Callable[
            [int, np.ndarray], np.ndarray
        ] = _get_new_layer_params,
        recorder: RecorderFactory = _recorder,
        parameter_preprocessor: ParameterPreprocessor = None,
    ) -> None:
        """
        Args:
            inner_optimizer: optimizer used for optimization at each layer.
            min_layer: starting number of layers.
            max_layer: maximum number of layers, at which optimization should stop.

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
        super().__init__(
            ansatz,
            inner_optimizer,
            estimation_tasks_factory,
            cost_function_factory,
            recorder,
        )
        assert 0 <= min_layer <= max_layer
        assert n_layers_per_iteration > 0
        self.min_layer = min_layer
        self.max_layer = max_layer
        self.n_layers_per_iteration = n_layers_per_iteration
        self.parameters_initializer = parameters_initializer
        self._parameter_preprocessor = parameter_preprocessor

    def minimize(
        self,
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all the parameters from the grid.

        Args:
            cost_function: object representing cost function we want to minimize
            inital_params: initial parameters for the cost function

        """
        ansatz = copy.deepcopy(self._ansatz)
        ansatz.number_of_layers = self.min_layer

        nit = 0
        nfev = 0
        histories = defaultdict(list)
        histories["history"] = []

        for i in range(self.min_layer, self.max_layer + 1, self.n_layers_per_iteration):
            assert ansatz.number_of_layers == i

            if i != self.min_layer:
                number_of_new_params = ansatz.number_of_params - len(optimal_params)
                initial_params = self.parameters_initializer(
                    number_of_new_params, optimal_params
                )

            estimation_tasks_factory = self._estimation_tasks_factory(ansatz=ansatz)

            cost_function = self._cost_function_factory(
                estimation_tasks_factory=estimation_tasks_factory,
            )

            # TODO update docstrings in QAOA parameter initialization with
            # parameter preprocessor arg in __init__ instead of __call__
            if hasattr(self._parameter_preprocessor, "n_layers"):
                self._parameter_preprocessor.n_layers = i

            # TODO gradient is not carried through w/ preprocessor.
            # Needs some python wizardry :p
            def new_cost_function(
                parameters: np.ndarray,
            ) -> Union[float, ValueEstimate]:
                return (
                    cost_function(self._parameter_preprocessor(parameters))
                    if self._parameter_preprocessor
                    else cost_function(parameters)
                )

            if keep_history:
                new_cost_function = self._recorder(new_cost_function)
            layer_results = self._inner_optimizer.minimize(
                new_cost_function, initial_params, keep_history=False
            )

            optimal_params: np.ndarray = layer_results.opt_params
            ansatz.number_of_layers += self.n_layers_per_iteration

            nfev += layer_results.nfev
            nit += layer_results.nit
            if keep_history:
                histories["history"] += new_cost_function.history
                if hasattr(new_cost_function, "gradient"):
                    histories["gradient_history"] += new_cost_function.gradient.history

        del layer_results["history"]
        del layer_results["nit"]
        del layer_results["nfev"]

        return OptimizeResult(**layer_results, **histories, nfev=nfev, nit=nit)