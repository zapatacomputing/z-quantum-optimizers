import unittest
import numpy as np
from scipy.optimize import OptimizeResult
from zquantum.core.circuit import ParameterGrid
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from .grid_search import GridSearchOptimizer

class CMAESOptimizerTests(unittest.TestCase, OptimizerTests):

    def setUp(self):
        grid = ParameterGrid([[0.5, 1.5, 0.1], [0.5, 1.5, 0.1]])
        self.optimizers = [GridSearchOptimizer(grid)]


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
