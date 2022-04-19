################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import copy
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
            recorder: recorder object which defines how to store
                the optimization history.
        """
        super().__init__(recorder=recorder)
        self.learning_rate = learning_rate

        assert number_of_iterations > 0
        self.number_of_iterations = number_of_iterations

    def _minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all
        the parameters from the provided list of points.

        Note:
            This optimizer does not require evaluation of the cost function,
            but relies only on gradient evaluation. This means, that if we want to
            keep track of values of the cost functions for each iteration, we
            need to perform extra evaluations. Therefore using `keep_history=True`
            will add extra evaluations that are not necessary for
            the optimization process itself.

        Args:
            cost_function: object representing cost function we want to minimize
            inital_params: initial parameters for the cost function
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded. Using this will increase runtime,
                see note.

        """
        assert isinstance(cost_function, CallableWithGradient)

        current_parameters = copy.deepcopy(initial_params)
        for _ in range(self.number_of_iterations):
            gradients = cost_function.gradient(current_parameters)
            current_parameters = current_parameters - (self.learning_rate * gradients)
            if keep_history:
                final_value = cost_function(current_parameters)

        if not keep_history:
            final_value = cost_function(current_parameters)

        return optimization_result(
            opt_value=final_value,
            opt_params=current_parameters,
            nit=self.number_of_iterations,
            nfev=None,
            **construct_history_info(cost_function, keep_history),  # type: ignore
        )
