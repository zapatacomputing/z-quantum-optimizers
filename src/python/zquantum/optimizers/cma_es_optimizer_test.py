import pytest
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.core.interfaces.mock_objects import mock_cost_function
from .cma_es_optimizer import CMAESOptimizer


@pytest.fixture(scope="function", params=[{"options": {"sigma_0": 0.1}}])
def optimizer(request):
    return CMAESOptimizer(**request.param)


class TestCMAESOptimizer(OptimizerTests):
    always_records_history = True

    def test_cmaes_optimizer_cannot_be_initialized_with_empty_options(self):
        with pytest.raises(RuntimeError):
            CMAESOptimizer(options={})

    def test_cmaes_optimizer_cannot_be_initialized_without_options(self):
        # This test is superfluous, as the TypeError is independent from the
        # body of CMAESOptimizer.__init__ and is raised automatically by
        # Python because the below call does not match __init__'s signature.
        with pytest.raises(TypeError):
            CMAESOptimizer()

    def test_cmaes_specific_fields(self):
        results = CMAESOptimizer(
            options={"sigma_0": 0.1, "maxfevals": 99, "popsize": 5}
        ).minimize(mock_cost_function, initial_params=[0, 0])

        assert "cma_xfavorite" in results
        assert isinstance(results["cma_xfavorite"], list)
        assert len(results["history"]) == 100
