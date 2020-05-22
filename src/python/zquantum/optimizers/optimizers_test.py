import unittest
import numpy as np
from .scipy_optimizer import ScipyOptimizer
from .grid_search import GridSearchOptimizer
from .cma_es_optimizer import CMAESOptimizer
from zquantum.core.circuit import ParameterGrid
from zquantum.core.cost_function import BasicCostFunction
from scipy.optimize import OptimizeResult

def rosen(x):
     """The Rosenbrock function"""
     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

class TestOptimizers(unittest.TestCase):

    def setUp(self):
        grid = ParameterGrid([[0.5, 1.5, 0.1], [0.5, 1.5, 0.1]])
        self.cost_function = BasicCostFunction(rosen)
        self.scipy_optimizer = ScipyOptimizer(method="L-BFGS-B")
        self.grid_search_optimizer = GridSearchOptimizer(grid)
        self.cmaes_optimizer = CMAESOptimizer(options={"sigma_0": 0.1})
        self.optimizers = [self.scipy_optimizer, self.grid_search_optimizer, self.cmaes_optimizer]

    def test_optimization(self):
        for optimizer in self.optimizers:
            results = optimizer.minimize(self.cost_function, initial_params=[0, 0])
            self.assertAlmostEqual(results.opt_value, 0, places=5)
            self.assertAlmostEqual(results.opt_params[0], 1, places=4)
            self.assertAlmostEqual(results.opt_params[1], 1, places=4)

            self.assertIn("nfev", results.keys())
            self.assertIn("nit", results.keys())
            self.assertIn("opt_value", results.keys())
            self.assertIn("opt_params", results.keys())
            self.assertIn("history", results.keys())


    def test_cmaes_optimizer_incorrect_initialization(self):
        self.assertRaises(RuntimeError, lambda: CMAESOptimizer(options={}))
        self.assertRaises(TypeError, lambda: CMAESOptimizer())

    def test_get_values_grid(self):
        # Given
        param_ranges = [(0, 1.1, 0.5)]*2
        grid = ParameterGrid(param_ranges)
        grid_search_optimizer = GridSearchOptimizer(grid)
        optimization_results = OptimizeResult()
        history = [{'value': 0}, {'value': 1}, {'value': 2}, {'value': 3}, {'value': 4}, {'value': 5}, {'value': 6}, {'value': 7}, {'value': 8}]
        optimization_results['history'] = history
        correct_values_grid = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        # When
        values_grid = grid_search_optimizer.get_values_grid(optimization_results)

        # Then
        np.testing.assert_array_equal(values_grid, correct_values_grid)
