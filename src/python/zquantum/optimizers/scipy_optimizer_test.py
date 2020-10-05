import unittest
import numpy as np
from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.interfaces.functions import FunctionWithGradient
from zquantum.core.interfaces.optimizer_test import (
    OptimizerTests,
    rosenbrock_function,
    sum_x_squared,
)
from .scipy_optimizer import ScipyOptimizer


class ScipyOptimizerTests(unittest.TestCase, OptimizerTests):
    def setUp(self):
        self.optimizers = [
            ScipyOptimizer(method="BFGS"),
            ScipyOptimizer(method="L-BFGS-B"),
            ScipyOptimizer(method="Nelder-Mead"),
            ScipyOptimizer(method="SLSQP"),
        ]

    def test_SLSQP_with_equality_constraints(self):
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
        self.assertAlmostEqual(results.opt_value, target_value, places=3)
        np.testing.assert_array_almost_equal(
            results.opt_params, target_params, decimal=3
        )

    def test_SLSQP_with_inequality_constraints(self):
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
        self.assertNotAlmostEqual(
            results_without_constraints.opt_value, results_with_constraints.opt_value
        )
        self.assertGreaterEqual(np.sum(results_with_constraints.opt_params), 3)

    def test_optimizer_succeeds_on_cost_function_without_gradient(self):
        for optimizer in self.optimizers:
            cost_function = sum_x_squared

            results = optimizer.minimize(
                cost_function, initial_params=np.array([1, -1])
            )
            self.assertAlmostEqual(results.opt_value, 0, places=5)
            self.assertAlmostEqual(results.opt_params[0], 0, places=4)
            self.assertAlmostEqual(results.opt_params[1], 0, places=4)

            self.assertIn("nfev", results.keys())
            self.assertIn("nit", results.keys())
            self.assertIn("opt_value", results.keys())
            self.assertIn("opt_params", results.keys())
            self.assertIn("history", results.keys())
