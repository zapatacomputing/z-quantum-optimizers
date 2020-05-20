import unittest
import numpy as np
from scipy.optimize import OptimizeResult
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from .scipy_optimizer import ScipyOptimizer

class CMAESOptimizerTests(unittest.TestCase, OptimizerTests):

    def setUp(self):
        self.optimizers = [ScipyOptimizer(method="L-BFGS-B"), ScipyOptimizer(method="nelder-mead")]