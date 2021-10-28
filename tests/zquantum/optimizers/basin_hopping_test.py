import pytest
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.optimizers.basin_hopping import BasinHoppingOptimizer


@pytest.fixture(
    params=[
        {
            "niter": 3,
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "BFGS",
            },
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "L-BFGS-B",
                "options": {
                    "ftol": 1e-7,
                },
            },
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "Nelder-Mead",
                "options": {
                    "fatol": 1e-7,
                },
            },
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "SLSQP",
                "options": {
                    "ftol": 1e-7,
                },
            },
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "COBYLA",
                "options": {"maxiter": 1000000, "tol": 1e-7},
            },
        },
    ]
)
def optimizer(request):
    return BasinHoppingOptimizer(**request.param)


@pytest.fixture(params=[True, False])
def keep_history(request):
    return request.param


class TestBasinHoppingOptimizer(OptimizerTests):
    pass
