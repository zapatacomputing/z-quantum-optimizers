import unittest
import numpy as np
import os
from .utils import (
    ValueEstimate,
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
