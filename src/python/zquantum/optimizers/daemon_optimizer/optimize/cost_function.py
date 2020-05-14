from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.utils import save_list, load_value_estimate
from zquantum.core.circuit import save_circuit_template_params
import time
import io

class ProxyCostFunction(CostFunction):
    """
    TODO
    """
    def __init__(self, client, save_evaluation_history=True):
        self.client = client
        self.save_evaluation_history = save_evaluation_history
        self.evaluations_history = [{'optimization-evaluation-ids': []}]
        self.current_iteration = 0

    def evaluate(self, parameters):
        """
        TODO
        """
        value = self._evaluate(parameters)
        return value

    def _evaluate(self, parameters):
        """
        TODO
        """
        # Encode params to json string
        save_circuit_template_params(parameters, 'current_optimization_params.json')
        with open('current_optimization_params.json', 'r') as f:
            current_params_string = f.read()

        # POST params to proxy
        evaluation_id = self.client.post_argument_values(current_params_string)

        # SAVE ID to optimization result['history']
        self.evaluations_history[self.current_iteration]['optimization-evaluation-ids'].append(evaluation_id)

        # POST status to EVALUATING
        self.client.post_status("EVALUATING")

        # WAIT for status to be OPTIMIZING
        while self.client.get_status() != "OPTIMIZING":
            time.sleep(1)

        # GET cost function evaluation from proxy
        evaluation_string = self.client.get_evaluation_result(evaluation_id)
        value_estimate = load_value_estimate(io.StringIO(evaluation_string))

        return value_estimate.value

    def get_gradient(self, parameters):
        raise NotImplemented

    def get_numerical_gradient(self, parameters):
        raise NotImplemented

    def callback(self, parameters):
        """
        TODO
        """
        self.evaluations_history[self.current_iteration]['params'] = parameters

        print("\nFinsished Iteration: {}".format(self.current_iteration), flush=True)
        print("Current Parameters: {}".format(parameters), flush=True)

        # If getting the value history, perform an evaluation with current parameters
        if self.save_evaluation_history:
            self.evaluations_history[self.current_iteration]['value'] = cost_function.evaluate(parameters)

            print("Current Value: {}".format(
                self.evaluations_history[self.current_iteration]['value']), flush=True)
        
        print("Starting Next Iteration...", flush=True)

        # Update currrent_iteration index and add new blank history
        self.current_iteration += 1
        self.evaluations_history.append({'optimization-evaluation-ids': []})
