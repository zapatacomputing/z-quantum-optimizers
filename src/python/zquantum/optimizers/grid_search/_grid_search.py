################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import warnings
from typing import Callable, Optional, Union

import numpy as np
from scipy.optimize import OptimizeResult
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    construct_history_info,
    optimization_result,
)
from zquantum.core.typing import RecorderFactory

from ._parameter_grid import ParameterGrid


class GridSearchOptimizer(Optimizer):
    def __init__(
        self,
        grid: ParameterGrid,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            grid: object defining for which parameters we want do the evaluations
            recorder: recorder object which defines how to store
                the optimization history.
        """
        warnings.warn(
            "The GridSearchOptimizer will soon be deprecated in favor"
            "of the SearchPointsOptimizer.",
            DeprecationWarning,
        )
        super().__init__(recorder=recorder)
        self.grid = grid

    def _minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: Optional[np.ndarray] = None,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying
        all the parameters from the grid.

        Args:
            cost_function: object representing cost function we want to minimize
            inital_params: initial parameters for the cost function
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """
        if initial_params is not None and len(initial_params) != 0:
            Warning("Grid search doesn't use initial parameters,they will be ignored.")

        min_value = None
        nfev = 0

        for params in self.grid.params_list:
            value = cost_function(params)
            nfev += 1
            if min_value is None or value < min_value:
                min_value = value
                optimal_params = params

        return optimization_result(
            opt_value=min_value,
            opt_params=optimal_params,
            nfev=nfev,
            nit=None,
            **construct_history_info(cost_function, keep_history)  # type: ignore
        )

    def get_values_grid(self, optimization_results: OptimizeResult) -> np.ndarray:
        """Shapes the values from the optimization results into the shape of the grid.

        Args:
            optimization_results: an optimization results dictionary

        Returns:
            numpy.ndarray: the values obtained at each grid point,
                shaped to have the same dimensions as the mesh grid
        """
        values = np.array([step["value"] for step in optimization_results["history"]])
        return np.reshape(values, self.grid.params_meshgrid[0].shape)
