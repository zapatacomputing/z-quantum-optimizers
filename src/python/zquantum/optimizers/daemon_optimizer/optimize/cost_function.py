from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.utils import save_list, load_value_estimate, ValueEstimate
from zquantum.core.circuit import save_circuit_template_params
import time
import io


class ProxyCostFunction:
    """Cost function using a proxy.

    Args:
        client: a client for interacting with the proxy.
        epsilon: finite difference step size.

    Params:
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
    """

    def __init__(self, client, epsilon: float = 1e-5):
        self.client = client
        self.current_iteration = 0
        self.epsilon = epsilon
        self.gradient = finite_differences_gradient(self.__call__)

    def __call__(self, parameters) -> ValueEstimate:
        """Evaluates the value of the cost function for given parameters by communicating with client.

        Args:
            parameters (np.ndarray): parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        # Encode params to json string
        save_circuit_template_params(parameters, "current_optimization_params.json")
        with open("current_optimization_params.json", "r") as f:
            current_params_string = f.read()

        # POST params to proxy
        evaluation_id = self.client.post_argument_values(current_params_string)

        # POST status to EVALUATING
        self.client.post_status("EVALUATING")

        # WAIT for status to be OPTIMIZING
        while self.client.get_status() != "OPTIMIZING":
            time.sleep(1)

        # GET cost function evaluation from proxy
        evaluation_string = self.client.get_evaluation_result(evaluation_id)
        value_estimate = load_value_estimate(io.StringIO(evaluation_string))

        return value_estimate

    def callback(self, parameters):
        """Callback function to be executed by the optimizer.

        Args:
            parameters (np.ndarray): parameters for which the evaluation should occur

        Returns:
            None
        """
        print("\nFinsished Iteration: {}".format(self.current_iteration), flush=True)
        print("Current Parameters: {}".format(parameters), flush=True)

        # If getting the value history, perform an evaluation with current parameters

        print("Starting Next Iteration...", flush=True)

        # Update currrent_iteration index and add new blank history
        self.current_iteration += 1
