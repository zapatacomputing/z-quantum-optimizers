from zquantum.optimizers.grid_search import load_parameter_grid

from zquantum.core.serialization import save_array, load_array

from zquantum.core.wip.circuits.layouts import load_circuit_connectivity
from zquantum.core.measurement import load_expectation_values
from zquantum.core.estimation import estimate_expectation_values_by_averaging
from zquantum.core.openfermion import load_qubit_operator
from zquantum.core.utils import create_object, load_noise_model, load_list
from zquantum.core.serialization import (
    save_optimization_results,
    load_optimization_results,
)
import yaml
import numpy as np
from typing import Optional
import warnings


def optimize_variational_circuit(
    ansatz_specs,
    backend_specs,
    optimizer_specs,
    cost_function_specs,
    qubit_operator,
    initial_parameters="None",
    fixed_parameters="None",
    noise_model="None",
    device_connectivity="None",
    parameter_grid="None",
    constraint_operator="None",
    prior_expectation_values: Optional[str] = None,
    thetas=None,
):
    warnings.warn(
        "optimize_variational_circuit will be depreciated in favor of optimize_ansatz_based_cost_function in steps/optimize.py in z-quantum-core.",
        DeprecationWarning,
    )
    if initial_parameters != "None":
        initial_params = load_array(initial_parameters)
    else:
        initial_params = None

    if fixed_parameters != "None":
        fixed_params = load_array(fixed_parameters)
    else:
        fixed_params = None

    # Load qubit operator
    operator = load_qubit_operator(qubit_operator)

    if isinstance(ansatz_specs, str):
        ansatz_specs_dict = yaml.load(ansatz_specs, Loader=yaml.SafeLoader)
    else:
        ansatz_specs_dict = ansatz_specs
    if "WarmStartQAOAAnsatz" in ansatz_specs_dict["function_name"]:
        thetas = np.array(load_list(thetas))
        ansatz = create_object(
            ansatz_specs_dict, cost_hamiltonian=operator, thetas=thetas
        )
    elif "QAOA" in ansatz_specs_dict["function_name"]:
        ansatz = create_object(ansatz_specs_dict, cost_hamiltonian=operator)
    else:
        ansatz = create_object(ansatz_specs_dict)

    # Load parameter grid
    if parameter_grid != "None":
        grid = load_parameter_grid(parameter_grid)
    else:
        grid = None

    # Load optimizer specs
    if isinstance(optimizer_specs, str):
        optimizer_specs_dict = yaml.load(optimizer_specs, Loader=yaml.SafeLoader)
    else:
        optimizer_specs_dict = optimizer_specs
    if (
        grid is not None
        and optimizer_specs_dict["function_name"] == "GridSearchOptimizer"
    ):
        optimizer = create_object(optimizer_specs_dict, grid=grid)
    else:
        optimizer = create_object(optimizer_specs_dict)

    # Load backend specs
    if isinstance(backend_specs, str):
        backend_specs_dict = yaml.load(backend_specs, Loader=yaml.SafeLoader)
    else:
        backend_specs_dict = backend_specs
    if noise_model != "None":
        backend_specs_dict["noise_model"] = load_noise_model(noise_model)
    if device_connectivity != "None":
        backend_specs_dict["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )
    backend = create_object(backend_specs_dict)

    # Load cost function specs
    if isinstance(cost_function_specs, str):
        cost_function_specs_dict = yaml.load(
            cost_function_specs, Loader=yaml.SafeLoader
        )
    else:
        cost_function_specs_dict = cost_function_specs
    estimation_method_specs = cost_function_specs_dict.pop(
        "estimation_method_specs", None
    )

    if estimation_method_specs is not None:
        if isinstance(estimation_method_specs, str):
            estimation_method_specs = yaml.loads(estimation_method_specs)
        estimation_method = create_object(estimation_method_specs)
    else:
        estimation_method = estimate_expectation_values_by_averaging
    cost_function_specs_dict["estimation_method"] = estimation_method

    estimation_preprocessors_specs_list = cost_function_specs_dict.pop(
        "estimation_preprocessors_specs", None
    )
    if estimation_preprocessors_specs_list is not None:
        estimation_preprocessors = []
        for estimation_preprocessor_specs in estimation_preprocessors_specs_list:
            if isinstance(estimation_preprocessor_specs, str):
                estimation_preprocessor_specs = yaml.loads(
                    estimation_preprocessor_specs
                )
            estimation_preprocessors.append(
                create_object(estimation_preprocessor_specs)
            )
        cost_function_specs_dict["estimation_preprocessors"] = estimation_preprocessors

    cost_function_specs_dict["target_operator"] = operator
    cost_function_specs_dict["ansatz"] = ansatz
    cost_function_specs_dict["backend"] = backend
    cost_function_specs_dict["fixed_parameters"] = fixed_params
    cost_function = create_object(cost_function_specs_dict)

    if prior_expectation_values is not None:
        if isinstance(prior_expectation_values, str):
            cost_function.estimator.prior_expectation_values = load_expectation_values(
                prior_expectation_values
            )

    if constraint_operator != "None":
        constraint_op = load_qubit_operator(constraint_operator)
        constraints_cost_function_specs = yaml.load(
            cost_function_specs, Loader=yaml.SafeLoader
        )
        constraints_estimator_specs = constraints_cost_function_specs.pop(
            "estimator-specs", None
        )
        if constraints_estimator_specs is not None:
            constraints_cost_function_specs["estimator"] = create_object(
                constraints_estimator_specs
            )
        constraints_cost_function_specs["ansatz"] = ansatz
        constraints_cost_function_specs["backend"] = backend
        constraints_cost_function_specs["target_operator"] = constraint_op
        constraint_cost_function = create_object(constraints_cost_function_specs)
        constraint_cost_function_wrapper = (
            lambda params: constraint_cost_function.evaluate(params).value
        )
        constraint_functions = (
            {"type": "eq", "fun": constraint_cost_function_wrapper},
        )
        optimizer.constraints = constraint_functions

    opt_results = optimizer.minimize(cost_function, initial_params)

    save_optimization_results(opt_results, "optimization-results.json")
    save_array(opt_results.opt_params, "optimized-parameters.json")


# CMA-ES optimization returns `xbest`, the best solution evaluated, but one might want to have `xfavorite`, which is the current best estimate of the optimum
def extract_xfav_params_from_cma_es_opt_results(optimization_results):
    opt_results = load_optimization_results(optimization_results)
    save_array(np.array(opt_results.cma_xfavorite), "fav-params.json")
