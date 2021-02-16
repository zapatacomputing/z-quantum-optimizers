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
):
    for i in range(number_of_repeats):
        print("Repeat", i)
        optimize_variational_circuit_with_layerwise_optimizer(
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
