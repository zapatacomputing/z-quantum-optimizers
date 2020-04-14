from zquantum.core.circuit import build_ansatz_circuit
from qeopenfermion import evaluate_qubit_operator
import copy

#TODO: tests
def optimize_variational_circuit(ansatz, operator, initial_params,
    backend, optimizer):
    """Optimize a variational circuit.
        TODO
    """
    
    #TODO: Do we want to keep that or move it outside?
    def get_cost_function(target_operator):

        def cost_function(params):
            # Build the ansatz circuit
            circuit = build_ansatz_circuit(ansatz, params)
            # TODO: leave it as it is or change?
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

