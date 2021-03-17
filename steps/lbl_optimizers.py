from zquantum.core.circuit import (
    load_circuit_template_params,
    save_circuit_template_params,
    load_parameter_grid,
    load_circuit_connectivity,
)
from zquantum.core.openfermion import load_qubit_operator
from zquantum.core.utils import create_object, load_noise_model
from zquantum.core.serialization import (
    save_optimization_results,
    load_optimization_results,
)

from zquantum.optimizers import LayerwiseAnsatzOptimizer

import yaml
import numpy as np
import os
import json
import copy


def get_random_parameters(number_of_params, params_min_values, params_max_values):
    return np.array(
        [
            params_min_values[i]
            + random.random() * (params_max_values[i] - params_min_values[i])
            for i in range(number_of_params)
        ]
    )


def repeated_optimize_variational_circuit_with_layerwise_optimizer(
    ansatz_specs,
    backend_specs,
    optimizer_specs,
    cost_function_specs,
    qubit_operator,
    min_layer,
    max_layer,
    params_min_values,
    params_max_values,
    number_of_repeats,
    use_lbl=True,
):
    final_value = None
    final_results = None

    for i in range(number_of_repeats):
        print("Repeat", i)
        if use_lbl:
            opt_results = optimize_variational_circuit_with_layerwise_optimizer(
                copy.deepcopy(ansatz_specs),
                copy.deepcopy(backend_specs),
                copy.deepcopy(optimizer_specs),
                copy.deepcopy(cost_function_specs),
                qubit_operator,
                min_layer,
                max_layer,
                params_min_values,
                params_max_values,
            )
        else:
            initial_parameters = np.random.uniform(
                low=params_min_values, high=params_max_values, size=max_layer * 2
            )

            opt_results = optimize_variational_circuit(
                copy.deepcopy(ansatz_specs),
                copy.deepcopy(backend_specs),
                copy.deepcopy(optimizer_specs),
                copy.deepcopy(cost_function_specs),
                qubit_operator,
                initial_parameters=initial_parameters,
            )

        if final_value is None or opt_results.opt_value < final_value:
            final_results = opt_results
            final_value = opt_results.opt_value

        os.rename("optimization-results.json", f"optimization-results-{i}.json")
        os.rename("optimized-parameters.json", f"optimized-parameters-{i}.json")

    final_results_list = {}
    final_parameters_list = {}
    for i in range(number_of_repeats):
        results_file = open(f"optimization-results-{i}.json", "r")
        parameters_file = open(f"optimized-parameters-{i}.json", "r")
        final_results_list[i] = yaml.load(results_file, Loader=yaml.SafeLoader)
        final_parameters_list[i] = yaml.load(parameters_file, Loader=yaml.SafeLoader)

    with open("optimization-results-list.json", "w") as outfile:
        json.dump(final_results_list, outfile)
    with open("optimized-parameters-list.json", "w") as outfile:
        json.dump(final_parameters_list, outfile)

    save_optimization_results(final_results, "optimization-results.json")
    save_circuit_template_params(final_results.opt_params, "optimized-parameters.json")


def optimize_variational_circuit_with_layerwise_optimizer(
    ansatz_specs,
    backend_specs,
    optimizer_specs,
    cost_function_specs,
    qubit_operator,
    min_layer,
    max_layer,
    params_min_values,
    params_max_values,
):
    # Load qubit operator
    operator = load_qubit_operator(qubit_operator)

    if isinstance(ansatz_specs, str):
        ansatz_specs_dict = yaml.load(ansatz_specs, Loader=yaml.SafeLoader)
    else:
        ansatz_specs_dict = ansatz_specs

    if ansatz_specs_dict["function_name"] == "QAOAFarhiAnsatz":
        ansatz = create_object(ansatz_specs_dict, cost_hamiltonian=operator)
    else:
        ansatz = create_object(ansatz_specs_dict)

    # Load optimizer specs
    if isinstance(optimizer_specs, str):
        optimizer_specs_dict = yaml.load(optimizer_specs, Loader=yaml.SafeLoader)
    else:
        optimizer_specs_dict = optimizer_specs
    optimizer = create_object(optimizer_specs_dict)

    # Load backend specs
    if isinstance(backend_specs, str):
        backend_specs_dict = yaml.load(backend_specs, Loader=yaml.SafeLoader)
    else:
        backend_specs_dict = backend_specs

    backend = create_object(backend_specs_dict)

    # Load cost function specs
    if isinstance(cost_function_specs, str):
        cost_function_specs_dict = yaml.load(
            cost_function_specs, Loader=yaml.SafeLoader
        )
    else:
        cost_function_specs_dict = cost_function_specs
    estimator_specs = cost_function_specs_dict.pop("estimator-specs", None)
    if estimator_specs is not None:
        cost_function_specs_dict["estimator"] = create_object(estimator_specs)
    cost_function_specs_dict["target_operator"] = operator
    cost_function_specs_dict["ansatz"] = ansatz
    cost_function_specs_dict["backend"] = backend
    cost_function = create_object(cost_function_specs_dict)

    lbl_optimizer = LayerwiseAnsatzOptimizer(inner_optimizer=optimizer)
    opt_results = lbl_optimizer.minimize_lbl(
        cost_function,
        min_layer=min_layer,
        max_layer=max_layer,
        params_min_values=params_min_values,
        params_max_values=params_max_values,
    )

    save_optimization_results(opt_results, "optimization-results.json")
    save_circuit_template_params(opt_results.opt_params, "optimized-parameters.json")
    return opt_results


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

    if isinstance(ansatz_specs, str):
        ansatz_specs_dict = yaml.load(ansatz_specs, Loader=yaml.SafeLoader)
    else:
        ansatz_specs_dict = ansatz_specs
    if ansatz_specs_dict["function_name"] == "QAOAFarhiAnsatz":
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
    save_circuit_template_params(opt_results.opt_params, "optimized-parameters.json")
    return opt_results