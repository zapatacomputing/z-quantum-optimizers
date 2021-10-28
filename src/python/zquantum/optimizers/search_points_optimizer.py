from typing import List, Optional

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


class SearchPointsOptimizer(Optimizer):
    def __init__(
        self,
        parameter_values_list: List[np.ndarray],
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            parameter_values_list: list of parameter values to evaluate
            recorder: recorder object which defines how to store
                the optimization history.
        """
        super().__init__(recorder=recorder)
        assert len(parameter_values_list) > 0
        self.parameter_values_list = parameter_values_list

    def _minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: Optional[np.ndarray] = None,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying
        all the parameters from the provided list of points.

        Args:
            cost_function: object representing cost function we want to minimize
            inital_params: initial parameters for the cost function
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """
        if initial_params is not None and len(initial_params) != 0:
            Warning(
                "DiscreteParameterValuesSearch search doesn't use initial parameters, "
                "they will be ignored."
            )

        min_value = None
        optimal_params = None

        for parameter_values in self.parameter_values_list:
            value = cost_function(parameter_values)
            if min_value is None or value < min_value:
                min_value = value
                optimal_params = parameter_values

        return optimization_result(
            opt_value=min_value,
            opt_params=optimal_params,
            nfev=len(self.parameter_values_list),
            nit=None,
            **construct_history_info(cost_function, keep_history)
        )
