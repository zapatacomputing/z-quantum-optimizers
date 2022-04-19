################################################################################
#© Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Optional

import numpy as np
from zquantum.core.serialization import save_array
from zquantum.core.typing import Specs, Union
from zquantum.core.utils import load_from_specs
from zquantum.optimizers.grid_search import ParameterGrid
from zquantum.optimizers.grid_search import (
    build_uniform_param_grid as _build_uniform_param_grid,
)
from zquantum.optimizers.grid_search import load_parameter_grid, save_parameter_grid


def build_uniform_param_grid(
    ansatz_specs: Optional[Specs] = None,
    number_of_params_per_layer: Optional[int] = None,
    number_of_layers: int = 1,
    min_value: float = 0,
    max_value: float = 2 * np.pi,
    step: float = np.pi / 5,
):
    assert (ansatz_specs is None) != (number_of_params_per_layer is None)

    if ansatz_specs is not None:
        ansatz = load_from_specs(ansatz_specs)
        number_of_params = ansatz.number_of_params
    else:
        number_of_params = number_of_params_per_layer

    grid = _build_uniform_param_grid(
        number_of_params, number_of_layers, min_value, max_value, step
    )
    save_parameter_grid(grid, "parameter-grid.json")


def get_parameter_values_list_from_grid(grid: Union[str, ParameterGrid]):
    if isinstance(grid, str):
        grid = load_parameter_grid(grid)

    parameter_values_list = np.array(grid.params_list)

    save_array(parameter_values_list, "parameter-values-list.json")
