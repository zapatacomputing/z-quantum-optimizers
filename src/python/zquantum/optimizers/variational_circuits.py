from zquantum.core.circuit import build_ansatz_circuit
from qeopenfermion import evaluate_qubit_operator
import copy

def optimize_variational_circuit(ansatz, operator, initial_params,
    backend, optimizer):
    """
    Calculates optimal parameters for the variational circuit.

    Args:
        ansatz(zquantum.core.CircuitTemplate): ansatz for which we want to find the optimal parameters
        operator(openfermion.SymbolicOperator): qubit operator for which we do the optimization
        initial_params(numpy.ndarray): initial parameters of the ansatz.
        backend(zquantum.core.interfaces.QuantumBackend): backend to be used for simulation
        optimizer(zquantum.core.interfaces.Optimizer): optimizer to be used for finding the optimal parameters
    
    Returns:
        optimization_results(scipy.optimize.OptimizeResults): results of the optimization.
    """
    
    def get_cost_function(target_operator):

        def cost_function(params):
            # Build the ansatz circuit
            circuit = build_ansatz_circuit(ansatz, params)

            operator_no_coeff = copy.deepcopy(target_operator)
            for term in target_operator.terms:
                operator_no_coeff.terms[term] = 1
    
            expectation_values = backend.get_expectation_values(circuit, operator_no_coeff)
            value_estimate = evaluate_qubit_operator(target_operator, expectation_values)
            return value_estimate.value

        return cost_function
    
    cost_function = get_cost_function(operator)
    optimization_results = optimizer.minimize(cost_function, initial_params)
    return optimization_results

