import unittest
import numpy as np
from zquantum.core.interfaces.optimizer_test import OptimizerTests, rosen
from .cma_es_optimizer import CMAESOptimizer

class CMAESOptimizerTests(unittest.TestCase, OptimizerTests):

    def setUp(self):
        self.optimizers = [CMAESOptimizer(options={"sigma_0": 0.1})]

    def test_incorrect_initialization(self):
        self.assertRaises(RuntimeError, lambda: CMAESOptimizer(options={}))
        self.assertRaises(TypeError, lambda: CMAESOptimizer())

    def test_cmaes_specific_fields(self):
        results = CMAESOptimizer(options={"sigma_0": 0.1, "maxfevals": 99, "popsize": 5}).minimize(rosen, initial_params=[0, 0])

        self.assertIn("cma_xfavorite", results.keys())
        self.assertIsInstance(results["cma_xfavorite"], list)
        self.assertEqual(len(results["history"]), 100)

