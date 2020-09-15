from zquantum.optimizers.utils import load_optimization_results
from zquantum.core.circuit import save_circuit_template_params
import numpy as np

# def optimize_variational_circuit():

# CMA-ES optimization returns `xbest`, the best solution evaluated, but one might want to have `xfavorite`, which is the current best estimate of the optimum
def extract_xfav_params_from_cma_es_opt_results(optimization_results):
    opt_results = load_optimization_results(optimization_results)
    save_circuit_template_params(np.array(opt_results.cma_xfavorite), "fav-params.json")
