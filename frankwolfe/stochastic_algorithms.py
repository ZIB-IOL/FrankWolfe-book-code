# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 08:50:19 2022

@author: pccom
"""

import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import math
from numba import jit

from frankwolfe.auxiliary_functions import (
    perform_update,
    is_vextex_present,
    calculate_stepsize_DIPCG,
    max_min_vertex,
    max_min_vertex_quick_exit,
    step_size_class,
    stopping_criterion,
)
from frankwolfe.algorithms import frank_wolfe
from frankwolfe.algorithms import generic_algorithm_wrapper

from frankwolfe.objective_functions import projection_problem_function

def stochastic_frank_wolfe(
    function,
    feasible_region,
    x0=None,
    stopping_criteria=None,
    step_size_parameters={
        "type_step": "constant_step",
        "alpha_max": 1.0,
    },
    algorithm_parameters={
        "algorithm_type": "standard",
        "maintain_active_set": False,
        "sigma": 2.0,
        "return_points": False,
    },
    disable_tqdm=False,
    logging_functions=None,
):
    if "algorithm_type" in algorithm_parameters:
        if algorithm_parameters["algorithm_type"] == "variance_reduced_SPIDER":
            algorithm_parameters["L"] = function.largest_eigenvalue_hessian()
            algorithm_parameters["D"] = feasible_region.diameter()
            algorithm_parameters["w"] = x0.copy()
            algorithm_parameters["d"] = np.zeros(x0.shape)
            
            step_size = step_size_class("constant_step", {
                "alpha_max": 1.0,
                "additive_constant": 2,
                "multiplicative_constant": 2,
            })
            
            step_type_to_use = stochastic_variance_reduced_SPIDER_frank_wolfe_step
        elif algorithm_parameters["algorithm_type"] == "conditional_gradient_sliding":
            algorithm_parameters["L"] = function.largest_eigenvalue_hessian()
            algorithm_parameters["D"] = feasible_region.diameter()
            algorithm_parameters["y"] = x0.copy()
            
            step_size = step_size_class("constant_step", {
                "alpha_max": 1.0,
                "additive_constant": 3,
                "multiplicative_constant": 3,
            })
            
            step_type_to_use = stochastic_conditional_gradient_sliding_frank_wolfe_step
        elif algorithm_parameters["algorithm_type"] == "variance_reduced":
            algorithm_parameters["w"] = x0.copy()
            step_type_to_use = stochastic_variance_reduced_frank_wolfe_step
            
            step_size = step_size_class("constant_step", {
                "alpha_max": 1.0,
                "additive_constant": 2.0,
                "multiplicative_constant": 2.0,
            })
            
            
        elif algorithm_parameters["algorithm_type"] == "momentum":
            algorithm_parameters["d"] = np.zeros(x0.shape)
            
            step_size = step_size_class("constant_step", {
                "type_step": "constant_step",
                "alpha_max": 1.0,
                "additive_constant": 8,
                "multiplicative_constant": 2,
            })
            
            step_type_to_use = stochastic_momentum_frank_wolfe_step
        elif algorithm_parameters["algorithm_type"] == "one_sample":
            algorithm_parameters["d"] = np.zeros(x0.shape)
            algorithm_parameters["w"] = x0.copy()
            
            step_size = step_size_class("constant_step", {
                "alpha_max": 1.0,
                "additive_constant": 1,
                "multiplicative_constant": 1,
            })
            
            step_type_to_use = stochastic_one_sample_frank_wolfe_step
        elif algorithm_parameters["algorithm_type"] == "constant_batch_size":
            if "batch_size" not in algorithm_parameters:
                algorithm_parameters["batch_size"] = 1000
                
            step_size = step_size_class("constant_step", {
                "alpha_max": 1.0,
                "additive_constant": 3,
                "multiplicative_constant": 2,
            })
            # step_size_parameters = 
            algorithm_parameters["d"] = np.zeros(x0.shape)
            algorithm_parameters["alpha_val"] = np.zeros(function.len)
            step_type_to_use = stochastic_constant_batch_size_frank_wolfe_step
        else:
            algorithm_parameters["L"] = function.largest_eigenvalue_hessian()
            algorithm_parameters["D"] = feasible_region.diameter()
            
            step_size = step_size_class("constant_step", {
                "type_step": "constant_step",
                "alpha_max": 1.0,
                "additive_constant": 8,
                "multiplicative_constant": 2,
            })
            
            step_type_to_use = stochastic_frank_wolfe_step
    else:
        
        algorithm_parameters["L"] = function.largest_eigenvalue_hessian()
        algorithm_parameters["D"] = feasible_region.diameter()
        
        step_size = step_size_class("constant_step", {
            "type_step": "constant_step",
            "alpha_max": 1.0,
            "additive_constant": 8,
            "multiplicative_constant": 2,
        })
        
        step_type_to_use = stochastic_frank_wolfe_step

    # In case we want to output data with a specific frequency
    algorithm_parameters["current_SFOO"] = 0
    algorithm_parameters["current_LMO"] = 0

    # Some hyperparameters that are needed for some algorithms
    algorithm_parameters["maintain_active_set"] = False
    return generic_algorithm_wrapper(
        step_type_to_use,
        function,
        feasible_region,
        x0=x0,
        stopping_criteria=stopping_criteria,
        convex_decomposition=None,
        active_set=None,
        step_size=step_size,
        algorithm_parameters=algorithm_parameters,
        disable_tqdm=disable_tqdm,
        logging_functions=logging_functions,
    )


def stochastic_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    convex_decomposition,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    if "SFOO_calls" not in data:
        data["SFOO_calls"] = [0]
    number_stochastic_calls = int(
        np.ceil(
            (
                algorithm_parameters["sigma"]
                * (i + 2)
                / (algorithm_parameters["L"] * algorithm_parameters["D"])
            )
            ** 2
        )
    )
    stochastic_grad = function.stochastic_grad(x, number_stochastic_calls)
    v = feasible_region.linear_optimization_oracle(stochastic_grad)
    d = v - x
    alpha = step_size.compute_step_size(function, x, d, grad, i)
    x += alpha * d
    # Update the calls used, in case we are exporting data with a particular frequency
    algorithm_parameters["current_SFOO"] += number_stochastic_calls
    algorithm_parameters["current_LMO"] += 1
    if (
        i % algorithm_parameters["frequency_output"] == 0
        or i <= algorithm_parameters["min_initial_output"]
    ):
        data["SFOO_calls"].append(
            data["SFOO_calls"][-1] + algorithm_parameters["current_SFOO"]
        )
        data["LMO_calls"].append(
            data["LMO_calls"][-1] + algorithm_parameters["current_LMO"]
        )
        algorithm_parameters["current_SFOO"] = 0
        algorithm_parameters["current_LMO"] = 0
    return x, None, None


def stochastic_one_sample_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    convex_decomposition,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    if "SFOO_calls" not in data:
        data["SFOO_calls"] = [0]

    rho = 1.0 / (i + 1)
    r, s = function.stochastic_grad_1SFW(x, algorithm_parameters["w"])
    algorithm_parameters["d"] = (1 - rho) * (algorithm_parameters["d"] - s) + r
    v = feasible_region.linear_optimization_oracle(algorithm_parameters["d"])
    alpha = step_size.compute_step_size(function, x, v - x, grad, i)
    algorithm_parameters["w"] = x.copy()
    x = (1 - alpha) * x + alpha * v
    # Update the calls used, in case we are exporting data with a particular frequency
    algorithm_parameters["current_SFOO"] += 2
    algorithm_parameters["current_LMO"] += 1
    if (
        i % algorithm_parameters["frequency_output"] == 0
        or i <= algorithm_parameters["min_initial_output"]
    ):
        data["SFOO_calls"].append(
            data["SFOO_calls"][-1] + algorithm_parameters["current_SFOO"]
        )
        data["LMO_calls"].append(
            data["LMO_calls"][-1] + algorithm_parameters["current_LMO"]
        )
        algorithm_parameters["current_SFOO"] = 0
        algorithm_parameters["current_LMO"] = 0
    return x, None, None


def stochastic_momentum_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    convex_decomposition,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    if "SFOO_calls" not in data:
        data["SFOO_calls"] = [0]

    stochastic_grad = function.stochastic_grad(x, 1)
    rho = 4.0 / np.power(i + 9, 2.0 / 3.0)
    algorithm_parameters["d"] = (1 - rho) * algorithm_parameters[
        "d"
    ] + rho * stochastic_grad
    v = feasible_region.linear_optimization_oracle(algorithm_parameters["d"])
    alpha = step_size.compute_step_size(function, x, v - x, stochastic_grad, i)
    x = (1 - alpha) * x + alpha * v
    # Update the calls used, in case we are exporting data with a particular frequency
    algorithm_parameters["current_SFOO"] += 1
    algorithm_parameters["current_LMO"] += 1
    if (
        i % algorithm_parameters["frequency_output"] == 0
        or i <= algorithm_parameters["min_initial_output"]
    ):
        data["SFOO_calls"].append(
            data["SFOO_calls"][-1] + algorithm_parameters["current_SFOO"]
        )
        data["LMO_calls"].append(
            data["LMO_calls"][-1] + algorithm_parameters["current_LMO"]
        )
        algorithm_parameters["current_SFOO"] = 0
        algorithm_parameters["current_LMO"] = 0
    return x, None, None

def stochastic_conditional_gradient_sliding_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    convex_decomposition,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    if "SFOO_calls" not in data:
        data["SFOO_calls"] = [0]

    alpha = step_size.compute_step_size(function, x, v - x, grad, i)
    beta = 4.0 * algorithm_parameters["L"] / (i + 3.0)
    eta = (
        algorithm_parameters["L"]
        * algorithm_parameters["D"] ** 2
        / ((i + 1.0) * (i + 2.0))
    )
    number_stochastic_calls = int(
        np.ceil(
            algorithm_parameters["sigma"] ** 2
            * (i + 3) ** 3
            / (algorithm_parameters["L"] * algorithm_parameters["D"]) ** 2
        )
    )
    stochastic_grad = function.stochastic_grad(
        (1 - alpha) * x + alpha * algorithm_parameters["y"], number_stochastic_calls
    )
    auxiliary_function = projection_problem_function(x, stochastic_grad, beta)
    inner_problem_data = frank_wolfe(
        auxiliary_function,
        feasible_region,
        x0 = algorithm_parameters["y"],
        stopping_criteria=stopping_criterion({"timing": 600.0,"iterations":int(1.0e10), "frank_wolfe_gap":eta}),
        step_size=step_size_class("line_search"),
        algorithm_parameters={
            "algorithm_type": "standard",
            # "maximum_time": 600.0,
            # "maximum_iterations": int(1.0e10),
            # "stopping_frank_wolfe_gap": eta,
            "maintain_active_set": False,
            "return_points": False,
        },
        disable_tqdm=True,
    )
    algorithm_parameters["y"] = inner_problem_data["solution"]
    x = x + alpha * (algorithm_parameters["y"] - x)

    # Update the calls used, in case we are exporting data with a particular frequency
    algorithm_parameters["current_SFOO"] += number_stochastic_calls
    algorithm_parameters["current_LMO"] += 1
    if (
        i % algorithm_parameters["frequency_output"] == 0
        or i <= algorithm_parameters["min_initial_output"]
    ):
        data["SFOO_calls"].append(
            data["SFOO_calls"][-1] + algorithm_parameters["current_SFOO"]
        )
        data["LMO_calls"].append(
            data["LMO_calls"][-1] + algorithm_parameters["current_LMO"]
        )
        algorithm_parameters["current_SFOO"] = 0
        algorithm_parameters["current_LMO"] = 0
    return x, None, None


def stochastic_constant_batch_size_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    convex_decomposition,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    if "SFOO_calls" not in data:
        data["SFOO_calls"] = [0]

    algorithm_parameters["alpha_val"], stoch_vector = function.stochastic_grad_CSFW(
        x, algorithm_parameters["batch_size"], algorithm_parameters["alpha_val"]
    )
    # stoch_grad_counter += batch_size
    algorithm_parameters["d"] += stoch_vector
    v = feasible_region.linear_optimization_oracle(algorithm_parameters["d"])
    alpha = step_size.compute_step_size(function, x, v - x, grad, i)
    x = (1 - alpha) * x + alpha * v

    # Update the calls used, in case we are exporting data with a particular frequency
    algorithm_parameters["current_SFOO"] += algorithm_parameters["batch_size"]
    algorithm_parameters["current_LMO"] += 1
    if (
        i % algorithm_parameters["frequency_output"] == 0
        or i <= algorithm_parameters["min_initial_output"]
    ):
        data["SFOO_calls"].append(
            data["SFOO_calls"][-1] + algorithm_parameters["current_SFOO"]
        )
        data["LMO_calls"].append(
            data["LMO_calls"][-1] + algorithm_parameters["current_LMO"]
        )
        algorithm_parameters["current_SFOO"] = 0
        algorithm_parameters["current_LMO"] = 0
    return x, None, None


def stochastic_variance_reduced_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    convex_decomposition,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    if "SFOO_calls" not in data:
        data["SFOO_calls"] = [0]
    if math.log(i + 1, 2).is_integer():
        algorithm_parameters["w"] = x.copy()
        grad_diff = np.zeros(len(algorithm_parameters["w"]))
        grad = function.full_grad_stochastic(algorithm_parameters["w"])
        algorithm_parameters["current_SFOO"] += int(function.len)
    else:
        grad_diff = function.stochastic_grad_SPIDER(
            x, algorithm_parameters["w"], int(48 * (i + 2))
        )
        algorithm_parameters["current_SFOO"] += int(48 * (i + 2))
    v = feasible_region.linear_optimization_oracle(grad + grad_diff)
    d = v - x
    alpha = step_size.compute_step_size(function, x, d, grad, i)
    x += alpha * d
    # Update the calls used, in case we are exporting data with a particular frequency
    algorithm_parameters["current_LMO"] += 1
    if (
        i % algorithm_parameters["frequency_output"] == 0
        or i <= algorithm_parameters["min_initial_output"]
    ):
        data["SFOO_calls"].append(
            data["SFOO_calls"][-1] + algorithm_parameters["current_SFOO"]
        )
        data["LMO_calls"].append(
            data["LMO_calls"][-1] + algorithm_parameters["current_LMO"]
        )
        algorithm_parameters["current_SFOO"] = 0
        algorithm_parameters["current_LMO"] = 0
    return x, None, None


def stochastic_variance_reduced_SPIDER_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    convex_decomposition,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    if "SFOO_calls" not in data:
        data["SFOO_calls"] = [0]
    if i == 0 or math.log(i, 2).is_integer():
        algorithm_parameters["d"] = function.stochastic_grad(
            x,
            int(
                np.ceil(
                    (
                        algorithm_parameters["sigma"]
                        * (i + 1)
                        / (algorithm_parameters["L"] * algorithm_parameters["D"])
                    )
                    ** 2
                )
            ),
        )
        algorithm_parameters["current_SFOO"] += int(
            np.ceil(
                (
                    algorithm_parameters["sigma"]
                    * (i + 1)
                    / (algorithm_parameters["L"] * algorithm_parameters["D"])
                )
                ** 2
            )
        )
    else:
        algorithm_parameters["d"] += function.stochastic_grad_SPIDER(
            x, algorithm_parameters["w"], int(16 * (i + 2))
        )
        algorithm_parameters["current_SFOO"] += int(16 * (i + 2))
    alpha = step_size.compute_step_size(function, x, v - x, grad, i)
    algorithm_parameters["w"] = x.copy()
    x += alpha * (
        feasible_region.linear_optimization_oracle(algorithm_parameters["d"]) - x
    )
    # Update the calls used, in case we are exporting data with a particular frequency
    algorithm_parameters["current_LMO"] += 1
    if (
        i % algorithm_parameters["frequency_output"] == 0
        or i <= algorithm_parameters["min_initial_output"]
    ):
        data["SFOO_calls"].append(
            data["SFOO_calls"][-1] + algorithm_parameters["current_SFOO"]
        )
        data["LMO_calls"].append(
            data["LMO_calls"][-1] + algorithm_parameters["current_LMO"]
        )
        algorithm_parameters["current_SFOO"] = 0
        algorithm_parameters["current_LMO"] = 0
    return x, None, None