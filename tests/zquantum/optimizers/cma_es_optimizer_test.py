import pytest
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.core.interfaces.mock_objects import mock_cost_function
from zquantum.optimizers.cma_es_optimizer import CMAESOptimizer


@pytest.fixture(scope="function")
def optimizer():
    return CMAESOptimizer(sigma_0=0.1)


class TestCMAESOptimizer(OptimizerTests):
    always_records_history = True

    def test_cmaes_specific_fields(self):
        results = CMAESOptimizer(
            sigma_0=0.1, options={"maxfevals": 99, "popsize": 5}
        ).minimize(mock_cost_function, initial_params=[0, 0])

        assert "cma_xfavorite" in results
        assert isinstance(results["cma_xfavorite"], list)
        assert len(results["history"]) == 100
