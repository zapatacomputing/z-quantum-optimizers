from zquantum.core.utils import convert_dict_to_array, convert_array_to_dict, SCHEMA_VERSION
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
            raise Exception("Required field " + field + " not present in optimization results")

    for field in recommended_fields:
        if field not in optimization_results.keys():
            warnings.warn("Recommended field " + field + " not present in optimization results", Warning)


def load_optimization_results(file):
    """Load a dict from a file.
    Args:
        file (str or file-like object): the name of the file, or a file-like
            object.
    
    Returns:
        dict: the optimization results
    """

    if isinstance(file, str):
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(file)

    optimization_results = {}
    for key in data:
        if key != 'history':
            optimization_results[key] = data[key]

    if data.get('history'):
        optimization_results['history'] = []
        for step in data['history']:
            optimization_results['history'].append(
                {'value': step['value'], 'params': convert_dict_to_array(step['params'])})

    return OptimizeResult(optimization_results)


def save_optimization_results(optimization_results, filename):
    """Save a dict to a file.
    Args:
        optimization_results (dict): the dict to be saved
        filename (str): the name of the file
    """
    data = {}
    data['schema'] = SCHEMA_VERSION + '-optimization_result'
    
    for key in optimization_results:
        if key != 'history':
            if type(optimization_results[key]) == np.ndarray:
                data[key] = optimization_results[key].tolist()
            elif type(optimization_results[key]) == bytes:
                data[key] = optimization_results[key].decode("utf-8")
            elif type(optimization_results[key]).__module__ == np.__name__:
                data[key] = optimization_results[key].item()
            else:
                data[key] = optimization_results[key]

    if optimization_results.get('history'):
        data['history'] = []
        for step in optimization_results['history']:
            if 'optimization-evaluation-ids' in step.keys():
                evaluation = {'value': step.get('value'),
                                        'params': convert_array_to_dict(step['params']),
                                        'optimization-evaluation-ids': step['optimization-evaluation-ids']}
                if 'bitstring_distribution' in step.keys():
                    evaluation['bitstring_distribution'] = step['bitstring_distribution']

                data['history'].append(evaluation)
            else:
                evaluation = {'value': step.get('value'),
                                        'params': convert_array_to_dict(step['params'])}
                if 'bitstring_distribution' in step.keys():
                    evaluation['bitstring_distribution'] = step['bitstring_distribution']
                data['history'].append(evaluation)

    with open(filename, 'w') as f:
        f.write(json.dumps(data, indent=2))
