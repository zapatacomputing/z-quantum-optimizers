from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.core.interfaces.mock_objects import mock_cost_function
from zquantum.optimizers.gd_optimizer import GDOptimizer
import pytest


@pytest.fixture(scope="function")
def optimizer():
    return GDOptimizer(method='RMSprop')


@pytest.fixture(params=[True, False])
def keep_history(request):
    return request.param


class TestGDOptimizer(OptimizerTests):
    def test_gd_specific_fields(self):
        results = GDOptimizer(
            method='adam', options={'tol':None, 'lr':0.1,'maxiter':100}
        ).minimize(mock_cost_function, initial_params=[0, 0], keep_history=True)

        assert len(results["history"]) == 100
