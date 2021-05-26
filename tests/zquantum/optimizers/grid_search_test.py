import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.optimizers.grid_search import (
    GridSearchOptimizer,
    ParameterGrid,
    build_uniform_param_grid,
)


@pytest.fixture(params=[ParameterGrid([[0, 1.5, 0.1], [0, 1.5, 0.1]])])
def optimizer(request):
    return GridSearchOptimizer(request.param)


class TestGridSearchOptimizer(OptimizerTests):
    def test_get_values_grid(self):
        # Given
        param_ranges = [(0, 1.1, 0.5)] * 2
        grid = ParameterGrid(param_ranges)
        grid_search_optimizer = GridSearchOptimizer(grid)
        optimization_results = OptimizeResult()
        history = [
            {"value": 0},
            {"value": 1},
            {"value": 2},
            {"value": 3},
            {"value": 4},
            {"value": 5},
            {"value": 6},
            {"value": 7},
            {"value": 8},
        ]
        optimization_results["history"] = history
        correct_values_grid = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        # When
        values_grid = grid_search_optimizer.get_values_grid(optimization_results)

        # Then
        np.testing.assert_array_equal(values_grid, correct_values_grid)


class TestParameterGrid:
    def test_dict_io(self):
        # Given
        param_ranges = [(0, 1, 0.1)] * 2
        grid = ParameterGrid(param_ranges)

        # When
        grid_dict = grid.to_dict()
        recreated_grid = ParameterGrid.from_dict(grid_dict)

        # Then
        np.testing.assert_array_equal(recreated_grid.param_ranges, grid.param_ranges)

    def test_params_list(self):
        # Given
        param_ranges = [(0, 1, 0.5)] * 2
        grid = ParameterGrid(param_ranges)
        correct_params_list = [
            np.array([0, 0]),
            np.array([0, 0.5]),
            np.array([0.5, 0]),
            np.array([0.5, 0.5]),
        ]

        # When
        params_list = grid.params_list

        # Then
        np.testing.assert_array_equal(params_list, correct_params_list)

    def test_params_meshgrid(self):
        # Given
        param_ranges = [(0, 1, 0.5)] * 2
        grid = ParameterGrid(param_ranges)
        correct_params_meshgrid = [
            np.array([[0, 0], [0.5, 0.5]]),
            np.array([[0, 0.5], [0, 0.5]]),
        ]

        # When
        params_meshgrid = grid.params_meshgrid

        # Then
        np.testing.assert_array_equal(params_meshgrid, correct_params_meshgrid)

    def test_build_uniform_param_grid(self):
        # Given
        n_params_per_layer = 2

        # When
        grid = build_uniform_param_grid(
            n_params_per_layer,
            n_layers=1,
            min_value=0.0,
            max_value=2 * np.pi,
            step=np.pi / 5,
        )

        # Then
        for i in range(10):
            for j in range(10):
                assert grid.params_list[i + 10 * j][0] == pytest.approx(j * np.pi / 5)
                assert grid.params_list[i + 10 * j][1] == pytest.approx(i * np.pi / 5)
                assert grid.params_meshgrid[0][i, j] == pytest.approx(i * np.pi / 5)
                assert grid.params_meshgrid[1][i, j] == pytest.approx(j * np.pi / 5)