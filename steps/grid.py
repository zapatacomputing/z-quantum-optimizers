from typing import Optional
from zquantum.core.typing import Specs
import numpy as np
from zquantum.core.utils import load_from_specs
from zquantum.optimizers.grid_search import (
    build_uniform_param_grid as _build_uniform_param_grid,
    save_parameter_grid,
)

# Build uniform parameter grid
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
