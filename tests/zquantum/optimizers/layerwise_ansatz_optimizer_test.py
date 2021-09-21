from unittest import mock

import numpy as np
from zquantum.optimizers.layerwise_ansatz_optimizer import LayerwiseAnsatzOptimizer
from zquantum.optimizers.scipy_optimizer import ScipyOptimizer
from zquantum.core.interfaces.mock_objects import MockAnsatz
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.optimizer_test import NESTED_OPTIMIZER_CONTRACTS
from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.interfaces.functions import FunctionWithGradient

import pytest
from functools import partial


ansatz = MockAnsatz(1, 5)


@pytest.fixture(
    params=[
        {
            "ansatz": ansatz,
            "inner_optimizer": ScipyOptimizer("L-BFGS-B"),
            "min_layer": 1,
            "max_layer": 3,
        }
    ]
)
def optimizer(request):
    return LayerwiseAnsatzOptimizer(**request.param)


def cost_function_factory(ansatz):
    def cost_function(x):
        return sum(x ** 2) * ansatz.number_of_layers

    return cost_function


initial_params = np.array([1])


class TestLayerwiseAnsatzOptimizer:
    @pytest.mark.parametrize("contract", NESTED_OPTIMIZER_CONTRACTS)
    def test_if_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer, cost_function_factory, initial_params)

    def test_ansatz_is_not_modified_outside_of_minimize(self, optimizer):
        initial_number_of_layers = ansatz.number_of_layers
        _ = optimizer.minimize(cost_function_factory, initial_params=initial_params)
        assert ansatz.number_of_layers == initial_number_of_layers

    @pytest.mark.parametrize("max_layer", [2, 3, 4, 5])
    def test_length_of_parameters_in_history_increases(self, max_layer):
        min_layer = 1
        optimizer = LayerwiseAnsatzOptimizer(
            ansatz=ansatz,
            inner_optimizer=ScipyOptimizer("L-BFGS-B"),
            min_layer=min_layer,
            max_layer=max_layer,
        )
        opt_results = optimizer.minimize(
            cost_function_factory, initial_params=np.ones(min_layer), keep_history=True
        )
        assert len(opt_results.opt_params) == max_layer

    @pytest.mark.parametrize(
        "min_layer,max_layer,n_layers_per_iteration",
        [[1, 2, 1], [1, 5, 1], [100, 120, 1], [1, 5, 2], [1, 10, 4], [1, 10, 20]],
    )
    def test_parameters_are_properly_initialized_for_each_layer(
        self, min_layer, max_layer, n_layers_per_iteration
    ):
        def parameters_initializer(number_of_params, old_params):
            return np.random.uniform(-np.pi, np.pi, number_of_params)

        parameters_initializer = mock.Mock(wraps=parameters_initializer)
        optimizer = LayerwiseAnsatzOptimizer(
            ansatz=ansatz,
            inner_optimizer=ScipyOptimizer("L-BFGS-B"),
            min_layer=min_layer,
            max_layer=max_layer,
            n_layers_per_iteration=n_layers_per_iteration,
            parameters_initializer=parameters_initializer,
        )

        _ = optimizer.minimize(cost_function_factory, initial_params=np.ones(min_layer))
        assert (
            parameters_initializer.call_count
            == (max_layer - min_layer) // n_layers_per_iteration
        )

        for ((args, _kwrgs), i) in zip(
            parameters_initializer.call_args_list,
            range(
                min_layer + n_layers_per_iteration, max_layer, n_layers_per_iteration
            ),
        ):
            number_of_params, old_params = args
            assert number_of_params == i

    @pytest.mark.parametrize("min_layer,max_layer", [[-1, 2], [3, 2], [-5, -1]])
    def test_fails_for_invalid_min_max_layer(self, min_layer, max_layer):
        with pytest.raises(AssertionError):
            LayerwiseAnsatzOptimizer(
                ansatz=ansatz,
                inner_optimizer=ScipyOptimizer("L-BFGS-B"),
                min_layer=min_layer,
                max_layer=max_layer,
            )
