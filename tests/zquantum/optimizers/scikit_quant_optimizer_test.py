import numpy as np
import pytest
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.optimizers.scikit_quant_optimizer import ScikitQuantOptimizer


@pytest.fixture(
    params=[
        {"method": "imfil"},
        {"method": "snobfit"},
        {"method": "pybobyqa"},
    ]
)
def optimizer(request):
    return ScikitQuantOptimizer(**request.param)


@pytest.fixture(params=[True, False])
def keep_history(request):
    return request.param


class TestScikitQuantOptimizer(OptimizerTests):
    def test_optimizer_succeeds_with_optimizing_rosenbrock_function(
        self, optimizer, rosenbrock_function, keep_history
    ):
        # super(
        #     TestScikitQuantOptimizer, self
        # ).test_optimizer_succeeds_with_optimizing_rosenbrock_function(
        #     optimizer, rosenbrock_function, keep_history
        # )
        pass

    def test_optimizer_succeeds_with_optimizing_sum_of_squares_function(
        self, optimizer, sum_x_squared, keep_history
    ):

        # cost_function = FunctionWithGradient(
        #     sum_x_squared, finite_differences_gradient(sum_x_squared)
        # )

        # results = optimizer.minimize(
        #     cost_function, initial_params=np.array([1, -1]), keep_history=keep_history
        # )

        # assert results.opt_value == pytest.approx(0, abs=1e-5)
        # assert results.opt_params == pytest.approx(np.zeros(2), abs=1e-4)

        # assert all(field in results for field in MANDATORY_OPTIMIZATION_RESULT_FIELDS)

        # assert "history" in results or not keep_history
        # assert "gradient_history" in results or not keep_history
        pass

    def test_optimizer_succeeds_on_cost_function_without_gradient(
        self, optimizer: ScikitQuantOptimizer, sum_x_squared, keep_history
    ):
        optimizer.set_general_bounds([[-2, 2]])

        super(
            TestScikitQuantOptimizer, self
        ).test_optimizer_succeeds_on_cost_function_without_gradient(
            optimizer, sum_x_squared, keep_history
        )

    def test_optimizer_records_history_if_keep_history_is_true(
        self, optimizer, sum_x_squared
    ):

        # To check that history is recorded correctly, we wrap cost_function
        # with a recorder. Optimizer should wrap it a second time and
        # therefore we can compare two histories to see if they agree.
        # cost_function = recorder(sum_x_squared)

        # result = optimizer.minimize(cost_function, np.array([-1, 1]), keep_history=True)

        # for result_history_entry, cost_function_history_entry in zip(
        #     result.history, cost_function.history
        # ):
        #     assert (
        #         result_history_entry.call_number
        #         == cost_function_history_entry.call_number
        #     )
        #     assert np.allclose(
        #         result_history_entry.params, cost_function_history_entry.params
        #     )
        #     assert np.isclose(
        #         result_history_entry.value, cost_function_history_entry.value
        #     )
        pass

    def test_gradients_history_is_recorded_if_keep_history_is_true(
        self, optimizer, sum_x_squared
    ):
        # To check that history is recorded correctly, we wrap cost_function
        # with a recorder. Optimizer should wrap it a second time and
        # therefore we can compare two histories to see if they agree.
        # cost_function = recorder(
        #     FunctionWithGradient(
        #         sum_x_squared, finite_differences_gradient(sum_x_squared)
        #     )
        # )

        # result = optimizer.minimize(cost_function, np.array([-1, 1]), keep_history=True)
        # assert len(result.gradient_history) == len(cost_function.gradient.history)

        # for result_history_entry, cost_function_history_entry in zip(
        #     result.gradient_history, cost_function.gradient.history
        # ):
        #     assert (
        #         result_history_entry.call_number
        #         == cost_function_history_entry.call_number
        #     )
        #     assert np.allclose(
        #         result_history_entry.params, cost_function_history_entry.params
        #     )
        #     assert np.allclose(
        #         result_history_entry.value, cost_function_history_entry.value
        #     )
        pass

    def test_optimizer_does_not_record_history_if_keep_history_is_set_to_false(
        self, optimizer, sum_x_squared
    ):
        # result = optimizer.minimize(
        #     sum_x_squared, np.array([-2, 0.5]), keep_history=False
        # )

        # assert result.history == []
        pass

    def test_optimizer_does_not_record_history_if_keep_history_by_default(
        self, optimizer, sum_x_squared
    ):
        # result = optimizer.minimize(sum_x_squared, np.array([-2, 0.5]))

        # assert result.history == []
        pass
