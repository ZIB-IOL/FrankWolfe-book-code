# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:24:18 2020

@author: pccom
"""
import numpy as np
import sys, os, time, datetime
import pickle

sys.path.append("..")

from objective_functions import quadratic_stochastic
from feasible_regions import box_constraints
from algorithms import FW_vanilla
from stochastic_algorithms import (
    stochastic_frank_wolfe,
)
from auxiliary_functions import (
    stopping_criterion,
)

dimension = int(sys.argv[1])
radius = float(sys.argv[2])
lower_bound = -radius * np.ones(dimension)
upper_bound = radius * np.ones(dimension)
feasible_region = box_constraints(lower_bound, upper_bound)

xOpt = 20.0 * (np.random.rand(dimension) - 1.0)
sigma = float(sys.argv[3])
maximum_time = float(sys.argv[4])
maximum_iterations = int(sys.argv[5])
minimum_eigenvalue_hessian = 0.0
maximum_eigenvalue_hessian = 100.0
maximum_eigenvalue_hessian = float(sys.argv[6])

function = quadratic_stochastic(
    dimension,
    xOpt,
    Mu=minimum_eigenvalue_hessian,
    L=maximum_eigenvalue_hessian,
    sigma=sigma,
)
G = np.sqrt(dimension) * sigma * (radius + 1)

initial_point = feasible_region.initial_point()
active_set = [initial_point]
barycentric_coordinates = [1.0]

if np.max(np.abs(xOpt)) <= radius:
    f_val_opt = function.f(xOpt)
else:
    results_FW = FW_vanilla(
        initial_point,
        function,
        feasible_region,
        barycentric_coordinates,
        active_set,
        maximum_iterations,
        step_size_param={"type_step": "line_search"},
        return_points=False,
        max_time=maximum_time,
        frank_wolfe_gap_tolerance=1.0e-8,
        disable_tqdm=False,
    )
    f_val_opt = results_FW["function_eval"][-1]

stopping_criteria = stopping_criterion({"timing": maximum_time,"iterations": maximum_iterations, "frank_wolfe_gap":1.0e-4})


results_stochastic = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria=stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "standard",
        "sigma": G,
        "return_points": False,
    },
    disable_tqdm=False,
)

results_SCGS = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria=stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "conditional_gradient_sliding",
        "sigma": G,
        "return_points": False,
    },
    disable_tqdm=False,
)

results_stochastic_SPIDER = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria=stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "variance_reduced_SPIDER",
        "sigma": G,
        "return_points": False,
    },
    disable_tqdm=False,
)

results_stochastic_one_sample = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria = stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "one_sample",
        "maintain_active_set": False,
        "return_points": False,
        "frequency_output": 1000,
        "min_initial_output": 1000,
    },
    disable_tqdm=False,
)

results_stochastic_momentum = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria = stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "momentum",
        "maintain_active_set": False,
        "return_points": False,
        "frequency_output": 1000,
        "min_initial_output": 1000,
    },
    disable_tqdm=False,
)


experiment_details = {
    "dimension": dimension,
    "Mu": minimum_eigenvalue_hessian,
    "L": maximum_eigenvalue_hessian,
    "feasible_region": "box constraints",
    "radius": radius,
    "sigma": sigma,
    "maximum_iterations": maximum_iterations,
    "maximum_time": maximum_time,
    "unconstrained_opt": xOpt,
}

# Put the results in a picle object and then output
SFW_data = {
    "name": r"SFW",
    "stochastic_oracle_calls": results_stochastic["stochastic_oracle_calls"],
    "f_value": results_stochastic["function_eval"],
    "primal_gap": [y - f_val_opt for y in results_stochastic["function_eval"]],
    "dual_gap": results_stochastic["frank_wolfe_gap"],
    "time": results_stochastic["timing"],
}
MSFW_data = {
    "name": r"Momentum SFW",
    "stochastic_oracle_calls": results_stochastic_momentum["stochastic_oracle_calls"],
    "LMO_oracle_calls": results_stochastic_momentum["LMO_calls"],
    "f_value": results_stochastic_momentum["function_eval"],
    "primal_gap": [y - f_val_opt for y in results_stochastic_momentum["function_eval"]],
    "dual_gap": results_stochastic_momentum["frank_wolfe_gap"],
    "time": results_stochastic_momentum["timing"],
}
SPIDERFW = {
    "name": r"SPIDER FW",
    "stochastic_oracle_calls": results_stochastic_SPIDER["stochastic_oracle_calls"],
    "f_value": results_stochastic_SPIDER["function_eval"],
    "primal_gap": [y - f_val_opt for y in results_stochastic_SPIDER["function_eval"]],
    "dual_gap": results_stochastic_SPIDER["frank_wolfe_gap"],
    "time": results_stochastic_SPIDER["timing"],
}
SFW1_data = {
    "name": r"1-SFW",
    "stochastic_oracle_calls": results_stochastic_one_sample["stochastic_oracle_calls"],
    "LMO_oracle_calls": results_stochastic_one_sample["LMO_calls"],
    "f_value": results_stochastic_one_sample["function_eval"],
    "primal_gap": [
        y - f_val_opt for y in results_stochastic_one_sample["function_eval"]
    ],
    "dual_gap": results_stochastic_one_sample["frank_wolfe_gap"],
    "time": results_stochastic_one_sample["timing"],
}
SCGS = {
    "name": r"SCGS",
    "stochastic_oracle_calls": results_SCGS["stochastic_oracle_calls"],
    "f_value": results_SCGS["function_eval"],
    "primal_gap": [y - f_val_opt for y in results_SCGS["function_eval"]],
    "dual_gap": results_SCGS["frank_wolfe_gap"],
    "time": results_SCGS["timing"],
}

results = {
    "details": experiment_details,
    "SFW": SFW_data,
    "SFW (momentum)": MSFW_data,
    "SPIDER": SPIDERFW,
    "1-SFW": SFW1_data,
    "SCGS": SCGS,
}

ts = time.time()
timestamp = (
    datetime.datetime.fromtimestamp(ts)
    .strftime("%Y-%m-%d %H:%M:%S")
    .replace(" ", "-")
    .replace(":", "-")
)
# Output the results as a pickled object for later use.
filepath = os.path.join(
    os.getcwd(),
    "Data",
    "stoch_"
    + str(int(sigma))
    + "sig_"
    + str(dimension)
    + "dim_"
    + str(int(radius))
    + "rad_"
    + str(int(minimum_eigenvalue_hessian))
    + "mu_"
    + str(int(maximum_eigenvalue_hessian))
    + "L_"
    + str(timestamp)
    + "_v3.pickle",
)
with open(filepath, "wb") as f:
    pickle.dump(results, f)
