from zquantum.core.interfaces.optimizer_test import (
    OptimizerTests,
)
from zquantum.optimizers.simple_gradient_descent import SimpleGradientDescent
from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.interfaces.functions import function_with_gradient
import pytest
import numpy as np


@pytest.fixture(
    params=[
        {"learning_rate": 0.1, "number_of_iterations": 100},
        {"learning_rate": 0.15, "number_of_iterations": 100},
        {"learning_rate": 0.215242, "number_of_iterations": 100},
        {"learning_rate": 0.99, "number_of_iterations": 1000},
    ]
)
def optimizer(request):
    return SimpleGradientDescent(**request.param)


@pytest.fixture(params=[True, False])
def keep_history(request):
    return request.param


class TestSimpleGradientDescent(OptimizerTests):
    @pytest.fixture
    def sum_x_squared(self):
        def _sum_x_squared(x):
            return sum(x ** 2)

        return function_with_gradient(
            _sum_x_squared, finite_differences_gradient(_sum_x_squared)
        )

    def test_optimizer_succeeds_on_cost_function_without_gradient(
        self, optimizer, sum_x_squared
    ):
        pytest.xfail(
            """This test fails since TestSimpleGradientDescent requires cost_function to have gradient method"""
        )

    def test_fails_to_initialize_when_number_of_iterations_is_negative(self):
        with pytest.raises(AssertionError):
            SimpleGradientDescent(0.1, -1)

    def test_fails_to_minimize_when_cost_function_does_not_have_gradient_method(
        self, optimizer
    ):
        cost_function = lambda x: sum(x)
        with pytest.raises(AssertionError):
            optimizer.minimize(cost_function, np.array([0, 0]))
