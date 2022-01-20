import numpy as np
import pytest
from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.interfaces.functions import FunctionWithGradient
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.optimizers.scipy_optimizer import ScipyOptimizer


@pytest.fixture(
    params=[
        {"method": "BFGS"},
        {"method": "L-BFGS-B"},
        {"method": "Nelder-Mead"},
        {"method": "SLSQP"},
        {"method": "COBYLA", "options": {"maxiter": 50000, "tol": 1e-7}},
    ]
)
def optimizer(request):
    return ScipyOptimizer(**request.param)


@pytest.fixture(
    params=[
        {"method": "L-BFGS-B"},
        {"method": "Nelder-Mead"},
        {"method": "SLSQP"},
    ]
)
def optimizer_with_bounds(request):
    bounds = [(2, 3), (2, 3)]
    return ScipyOptimizer(bounds=bounds, **request.param)


@pytest.fixture(params=[True, False])
def keep_history(request):
    return request.param


class TestScipyOptimizer(OptimizerTests):
    def test_optimizers_work_with_bounds_provided(
        self, optimizer_with_bounds, sum_x_squared
    ):

        # Given
        cost_function = FunctionWithGradient(
            sum_x_squared, finite_differences_gradient(sum_x_squared)
        )

        initial_params = np.array([2.5, 2.5])
        target_params = np.array([2, 2])
        target_value = 8

        # When
        results = optimizer_with_bounds.minimize(
            cost_function, initial_params=initial_params
        )

        # Then
        assert results.opt_value == pytest.approx(target_value, abs=1e-3)
        assert results.opt_params == pytest.approx(target_params, abs=1e-3)

    def test_SLSQP_with_equality_constraints(self, sum_x_squared, rosenbrock_function):
        # Given
        cost_function = FunctionWithGradient(
            rosenbrock_function, finite_differences_gradient(rosenbrock_function)
        )
        constraint_cost_function = sum_x_squared

        constraints = ({"type": "eq", "fun": constraint_cost_function},)
        optimizer = ScipyOptimizer(method="SLSQP", constraints=constraints)
        initial_params = np.array([1, 1])
        target_params = np.array([0, 0])
        target_value = 1

        # When
        results = optimizer.minimize(cost_function, initial_params=initial_params)

        # Then
        assert results.opt_value == pytest.approx(target_value, abs=1e-3)
        assert results.opt_params == pytest.approx(target_params, abs=1e-3)

    def test_SLSQP_with_inequality_constraints(self, rosenbrock_function):
        # Given
        cost_function = FunctionWithGradient(
            rosenbrock_function, finite_differences_gradient(rosenbrock_function)
        )
        constraints = {"type": "ineq", "fun": lambda x: x[0] + x[1] - 3}
        optimizer = ScipyOptimizer(method="SLSQP")
        initial_params = np.array([0, 0])

        # When
        results_without_constraints = optimizer.minimize(
            cost_function, initial_params=initial_params
        )
        optimizer.constraints = constraints
        results_with_constraints = optimizer.minimize(
            cost_function, initial_params=initial_params
        )

        # Then
        assert results_without_constraints.opt_value == pytest.approx(
            results_with_constraints.opt_value, abs=1e-1
        )
        assert results_with_constraints.opt_params.sum() >= 3
