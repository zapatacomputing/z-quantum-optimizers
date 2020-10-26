from zquantum.core.utils import (
    ValueEstimate,
    convert_dict_to_array,
    convert_array_to_dict,
    SCHEMA_VERSION,
)
from scipy.optimize import OptimizeResult
import json
import warnings
import numpy as np


def validate_optimization_results(optimization_results):
    """
    Validates optimization results.
    Raises exception if required fields are missing.
    Raises warning if recommended fields are missing

    Args:
        optimization_results(OptimizeResults): return object of the optimizer.
    """
    required_fields = ["opt_value", "opt_params"]
    recommended_fields = ["nfev", "nit"]
    for field in required_fields:
        if field not in optimization_results.keys():
            raise Exception(
                "Required field " + field + " not present in optimization results"
            )

    for field in recommended_fields:
        if field not in optimization_results.keys():
            warnings.warn(
                "Recommended field " + field + " not present in optimization results",
                Warning,
            )
