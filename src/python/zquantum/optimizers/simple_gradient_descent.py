from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    optimization_result,
    construct_history_info,
)
from zquantum.core.typing import RecorderFactory

from scipy.optimize import OptimizeResult
from typing import Optional
import numpy as np
import copy


class SimpleGradientDescent(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        number_of_iterations: int,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            parameter_values_list: list of parameter values to evaluate
            recorder: recorder object which defines how to store the optimization history.
        """
        super().__init__(recorder=recorder)
        self.learning_rate = learning_rate

        assert number_of_iterations > 0
        self.number_of_iterations = number_of_iterations

    def _minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: Optional[np.ndarray] = None,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all the parameters from the provided list of points.

        Args:
            cost_function: object representing cost function we want to minimize
            inital_params: initial parameters for the cost function
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """
        assert hasattr(cost_function, "gradient")

        current_parameters = copy.deepcopy(initial_params)
        for _ in range(self.number_of_iterations):
            gradients = cost_function.gradient(current_parameters)
            current_parameters = current_parameters - (self.learning_rate * gradients)

        final_value = cost_function(current_parameters)

        return optimization_result(
            opt_value=final_value,
            opt_params=current_parameters,
            nit=self.number_of_iterations,
            nfev=None,
            **construct_history_info(cost_function, keep_history),
        )
