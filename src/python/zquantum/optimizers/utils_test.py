import unittest
import numpy as np
import os
from .utils import (
    load_optimization_results,
    save_optimization_results,
    validate_optimization_results,
)
from zquantum.core.utils import convert_dict_to_array, convert_array_to_dict
from scipy.optimize import OptimizeResult
from scipy.optimize import OptimizeResult
import warnings


class TestUtils(unittest.TestCase):
    def test_validate_optimization_results(self):
        opt_results = OptimizeResult({})

        with self.assertRaises(Exception):
            validate_optimization_results(opt_results)

        opt_results["opt_value"] = 1
        opt_results["opt_params"] = [1]

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Trigger a warning.
            validate_optimization_results(opt_results)

            # Verify some things
            assert len(w) == 2
            assert "nfev" in str(w[-2].message)
            assert "nit" in str(w[-1].message)

        opt_results["nfev"] = 5
        opt_results["nit"] = 2

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Trigger a warning.
            validate_optimization_results(opt_results)

            # Verify some things
            assert len(w) == 0

    def test_optimization_result_io(self):
        opt_results = OptimizeResult({})
        opt_results["value"] = 0.1
        opt_results["status"] = 0
        opt_results["success"] = True
        opt_results["nfev"] = 5
        opt_results["nit"] = 2
        opt_results["history"] = [
            {"value": 0.1, "params": np.array([1.0, 2.0])},
            {"value": -0.1, "params": np.array([-1.0, 2.0])},
        ]
        opt_results["opt_value"] = 5
        opt_results["opt_params"] = np.array([1.0, 2.0])

        save_optimization_results(opt_results, "opt_result.json")
        loaded_result = load_optimization_results("opt_result.json")

        for key in opt_results:
            if key != "history" and key != "opt_params":
                self.assertEqual(opt_results[key], loaded_result[key])

        self.assertTrue(
            np.allclose(opt_results["opt_params"], loaded_result["opt_params"])
        )
        # Verify that the optimization history is the same
        self.assertEqual(len(opt_results["history"]), len(loaded_result["history"]))
        for i in range(len(opt_results["history"])):
            self.assertAlmostEqual(
                opt_results["history"][i]["value"], loaded_result["history"][i]["value"]
            )
            self.assertTrue(
                np.allclose(
                    opt_results["history"][i]["params"],
                    loaded_result["history"][i]["params"],
                )
            )

        os.remove("opt_result.json")
