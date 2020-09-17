from zquantum.optimizers.utils import (
    load_optimization_results,
    save_optimization_results,
)
from zquantum.core.circuit import (
    load_circuit_template_params,
    save_circuit_template_params,
    load_parameter_grid,
    load_circuit_connectivity,
)
from qeopenfermion import load_qubit_operator
from zquantum.core.utils import create_object, load_noise_model
import json
import numpy as np


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
):
    if initial_parameters != "None":
        initial_params = load_circuit_template_params(initial_parameters)
    else:
        initial_params = None

    if fixed_parameters != "None":
        fixed_params = load_circuit_template_params(fixed_parameters)
    else:
        fixed_params = None

    # Load qubit operator
    operator = load_qubit_operator(qubit_operator)

    ansatz_specs_dict = json.loads(ansatz_specs)
    if ansatz_specs_dict["function_name"] == "QAOAFarhiAnsatz":
        ansatz = create_object(ansatz_specs_dict, cost_hamiltonian=operator)
    else:
        ansatz = create_object(ansatz_specs_dict)

    # Load parameter grid
    if parameter_grid != "None":
        grid = load_parameter_grid(parameter_grid)
    else:
        grid = None

    optimizer_specs_dict = json.loads(optimizer_specs)
    if (
        grid is not None
        and optimizer_specs_dict["function_name"] == "GridSearchOptimizer"
    ):
        optimizer = create_object(optimizer_specs_dict, grid=grid)
    else:
        optimizer = create_object(optimizer_specs_dict)

    backend_specs_dict = json.loads(backend_specs)
    if noise_model != "None":
        backend_specs_dict["noise_model"] = load_noise_model(noise_model)
    if device_connectivity != "None":
        backend_specs_dict["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )
    backend = create_object(backend_specs_dict)

    cost_function_specs_dict = json.loads(cost_function_specs)
    estimator_specs = cost_function_specs_dict.pop("estimator-specs", None)
    if estimator_specs is not None:
        cost_function_specs_dict["estimator"] = create_object(estimator_specs)
    cost_function_specs_dict["target_operator"] = operator
    cost_function_specs_dict["ansatz"] = ansatz
    cost_function_specs_dict["backend"] = backend
    cost_function_specs_dict["fixed_parameters"] = fixed_params
    cost_function = create_object(cost_function_specs_dict)

    if constraint_operator != "None":
        constraint_op = load_qubit_operator(constraint_operator)
        constraints_cost_function_specs = json.loads(cost_function_specs)
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

    print(opt_results)
    print(opt_results.opt_params)

    save_optimization_results(opt_results, "optimization-results.json")
    print("Saved opt results")
    save_circuit_template_params(opt_results.opt_params, "optimized-parameters.json")
    print("saved opt params")


# CMA-ES optimization returns `xbest`, the best solution evaluated, but one might want to have `xfavorite`, which is the current best estimate of the optimum
def extract_xfav_params_from_cma_es_opt_results(optimization_results):
    opt_results = load_optimization_results(optimization_results)
    save_circuit_template_params(np.array(opt_results.cma_xfavorite), "fav-params.json")
