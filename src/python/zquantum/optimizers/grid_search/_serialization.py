from ._parameter_grid import ParameterGrid
from zquantum.core.utils import SCHEMA_VERSION
from zquantum.core.typing import Writeable
import json


def save_parameter_grid(grid: ParameterGrid, filename: str) -> None:
    """Saves a parameter grid to a file.

    Args:
        grid (core.circuit.ParameterGrid): the parameter grid to be saved
        filename (str): the name of the file
    """

    data = grid.to_dict()
    data["schema"] = SCHEMA_VERSION + "-parameter_grid"

    with open(filename, "w") as f:
        f.write(json.dumps(data))


def load_parameter_grid(file: Writeable) -> ParameterGrid:
    """Loads a parameter grid from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        core.circuit.ParameterGrid: the parameter grid
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return ParameterGrid.from_dict(data)
