from zquantum.core.circuit import save_circuit_template_params
from zquantum.core.utils import (
    load_value_estimate,
    convert_array_to_dict,
    SCHEMA_VERSION,
)
from .cost_function import ProxyCostFunction
import scipy
import json

# ---------- Optimize Variational Circuit ----------
def optimize_variational_circuit_with_proxy(
    initial_params, optimizer, client, **kwargs
):
    """Optimizes a variational circuit using proxy architecture.
    
    Arguments:
        initial_params (numpy.ndarray): initial guess for the ansatz parameters.
        method (string): scipy method for optimization
        client (zquantum.core.optimizer.proxy.Client): a client for interacting with
            the proxy

        *** OPTIONAL ***
        keep_value_history (bool): If true, an evaluation is done after every
            iteration of the optimizer and the value is saved
        layers_to_optimize (str): which layers of the ansatz to optimize. Options
            are 'all' and 'last'.
        options (dict): options for scipy optimizer
        **kwargs: keyword arguments passed to zquantum.core.optimization.minimize

    Returns:
        tuple: two-element tuple containing

        - **results** (**dict**): a dictionary with keys `value`, `status`,
                `success`, `nfev`, and `nit`
        - **optimized_params** (**numpy.array**): the optimized parameters
    """
    # Define cost function that interacts with client
    cost_function = ProxyCostFunction(client)

    # POST status to OPTIMIZING
    client.post_status("OPTIMIZING")

    # Perform the minimization
    opt_results = optimizer.minimize(
        cost_function, initial_params, callback=cost_function.callback
    )

    # Update opt_results object
    # TODO: this is done temporarily to ensure no data is lost. However, storing history
    # should be handled by optimizer, not cost_function.
    opt_results["history"] = cost_function.evaluations_history

    # Since a new history element is added in the callback function, if there is
    #   at least one iteration, there is an empty history element at the end
    #   that must be removed
    if "nit" in opt_results.keys() and opt_results.nit > 0:
        del opt_results["history"][-1]

    # POST status to DONE
    client.post_status("DONE")

    return opt_results
