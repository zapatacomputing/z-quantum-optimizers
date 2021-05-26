import os
from zquantum.optimizers.serialization import save_parameter_grid, load_parameter_grid
from zquantum.optimizers.grid_search import ParameterGrid


def test_parameter_grid_io():
    # Given
    param_ranges = [(0, 1, 0.1)] * 2
    grid = ParameterGrid(param_ranges)

    # When
    save_parameter_grid(grid, "grid.json")
    loaded_grid = load_parameter_grid("grid.json")

    # Then
    assert len(grid.param_ranges) == len(loaded_grid.param_ranges)
    for i in range(len(grid.param_ranges)):
        assert tuple(grid.param_ranges[i]) == tuple(loaded_grid.param_ranges[i])
    os.remove("grid.json")
