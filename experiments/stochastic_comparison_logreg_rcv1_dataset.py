# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:24:18 2020

@author: pccom
"""
import numpy as np
import sys, os, time, datetime
import copt as cp
import pickle

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


from frankwolfe.feasible_regions import L1_ball
from frankwolfe.auxiliary_functions import smoothness_estimate
from frankwolfe.objective_functions import logistic_regression
from frankwolfe.stochastic_algorithms import (
    stochastic_frank_wolfe,
)
from frankwolfe.auxiliary_functions import (
    stopping_criterion,
)

A, y = cp.datasets.load_rcv1("full")
m, n = A.shape
feasible_region = L1_ball(n, alpha=100.0)
function = logistic_regression(A, y, L=0.0005634358252303922)

line_search = {
    "type_step": "adaptive_short_step",
    "L_estimate": smoothness_estimate(feasible_region.initial_point(), function),
    "eta": 0.9,
    "tau": 2.0,
}
initial_point = feasible_region.initial_point()

maximum_time = 28800
maximum_iterations = 5000000000000
G = np.sqrt(0.12961177486342798)  # 0.01
target_FW_gap = 1.0e-8
initial_point = feasible_region.initial_point()
f_val_opt = 0.4283901598507915

stopping_criteria = stopping_criterion({"timing": maximum_time,"iterations": maximum_iterations, "frank_wolfe_gap":target_FW_gap})

print("Running CSFW.")
results_stochastic_constant_batch_size = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria=stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "constant_batch_size",
        "batch_size": 1,
        "return_points": False,
        "frequency_output": 100,
        "min_initial_output": 1000,
    },
    disable_tqdm=False,
)


print("Running SVRFW.")
results_SVRFW = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria = stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "variance_reduced",
        "maintain_active_set": False,
        "sigma": G,
        "return_points": False,
    },
    disable_tqdm=False,
)


print("Running SFW.")
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

print("Running SCGS.")
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

print("Running 1-SFW.")
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

print("Running M-SFW.")
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

print("Running SPIDER.")
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

experiment_details = {
    "dimension": n,
    "numbersamples": m,
    "G": G,
    "maximum_iterations": maximum_iterations,
    "maximum_time": maximum_time,
}

# Put the results in a picle object and then output
SVRFW_data = {
    "name": r"SVRFW_data",
    "stochastic_oracle_calls": results_SVRFW["stochastic_oracle_calls"],
    "f_value": results_SVRFW["function_eval"],
    "primal_gap": [y - f_val_opt for y in results_SVRFW["function_eval"]],
    "dual_gap": results_SVRFW["frank_wolfe_gap"],
    "time": results_SVRFW["timing"],
}
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

CSFW_data = {
    "name": r"CSFW",
    "stochastic_oracle_calls": results_stochastic_constant_batch_size[
        "stochastic_oracle_calls"
    ],
    "f_value": results_stochastic_constant_batch_size["function_eval"],
    "primal_gap": [
        y - f_val_opt for y in results_stochastic_constant_batch_size["function_eval"]
    ],
    "dual_gap": results_stochastic_constant_batch_size["frank_wolfe_gap"],
    "time": results_stochastic_constant_batch_size["timing"],
}
SFW1_data = {
    "name": r"1-SFW",
    "stochastic_oracle_calls": results_stochastic_one_sample["stochastic_oracle_calls"],
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
    "SVRFW": SVRFW_data,
    "SFW": SFW_data,
    "SFW (momentum)": MSFW_data,
    "SPIDER": SPIDERFW,
    "1-SFW": SFW1_data,
    "SCGS": SCGS,
    "CSFW": CSFW_data,
}

output_directory = os.path.join(os.getcwd(), "output_data")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
    
output_directory_images = os.path.join(os.getcwd(), "output_images")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory_images):
    os.makedirs(output_directory_images)

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
    "output_data",
    "stoch_logreg_rcv1_dataset.pickle",
)
with open(filepath, "wb") as f:
    pickle.dump(results, f)
    
output_filepath = os.path.join(
    output_directory,
    "logreg_rcv1_FOO.pdf",
)

list_x = [
    np.asarray(results["SFW"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["SPIDER"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["SFW (momentum)"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["1-SFW"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["SCGS"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["CSFW"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["SVRFW"]["stochastic_oracle_calls"]) + 1,
]
list_data = [
    results["SFW"]["primal_gap"],
    results["SPIDER"]["primal_gap"],
    results["SFW (momentum)"]["primal_gap"],
    results["1-SFW"]["primal_gap"],
    results["SCGS"]["primal_gap"],
    results["CSFW"]["primal_gap"],
    results["SVRFW"]["primal_gap"],
]
list_legend = [
    "SFW",
    "SPIDER",
    "M-SFW",
    "1-SFW",
    "SCGS",
    "CSFW",
    "SVRF",
]
list_legend = [
    r"$\mathrm{SFW}$",
    r"$\mathrm{SPIDER}$",
    r"$\mathrm{M-SFW}$",
    r"$\mathrm{1-SFW}$",
    r"$\mathrm{SCGS}$",
    r"$\mathrm{CSFW}$",
    r"$\mathrm{SVRF}$",
]


x_limits = [1, None]

figure_size = 7.3
from frankwolfe.plotting_function import plot_results

colors_fill = ["None", "None", "None", "None", "None", "None", "None"]
colors = ["k", "b", "m", "c", "r", "g", "y"]
markers = ["o", "s", "^", "P", "X", "H", "D"]

plot_results(
    list_x,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Number \ of \ SFOO \ calls}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=x_limits,
    save_figure=output_filepath,
    figure_width=figure_size,
)

output_filepath = os.path.join(
    output_directory,
    "logreg_rcv1_LMO_v7.pdf",
)

list_x = [
    np.arange(len(results["SFW"]["primal_gap"])) + 1,
    np.arange(len(results["SPIDER"]["primal_gap"])) + 1,
    np.arange(len(results["SFW (momentum)"]["primal_gap"])) + 1,
    np.arange(len(results["1-SFW"]["primal_gap"])) + 1,
    np.arange(len(results["SCGS"]["primal_gap"])) + 1,
    np.arange(len(results["CSFW"]["primal_gap"])) + 1,
    np.arange(len(results["SVRFW"]["primal_gap"])) + 1,
]

plot_results(
    list_x,
    list_data,
    [],
    "",
    r"$\mathrm{Number \ of \ LMO \ calls}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=x_limits,
    save_figure=output_filepath,
    figure_width=figure_size,
)

output_filepath = os.path.join(
    output_directory,
    "logreg_rcv1_time_v7.pdf",
)

list_x = [
    results["SFW"]["time"],
    results["SPIDER"]["time"],
    results["SFW (momentum)"]["time"],
    results["1-SFW"]["time"],
    results["SCGS"]["time"],
    results["CSFW"]["time"],
    results["SVRFW"]["time"],
]

list_data = [
    results["SFW"]["primal_gap"],
    results["SPIDER"]["primal_gap"],
    results["SFW (momentum)"]["primal_gap"],
    results["1-SFW"]["primal_gap"],
    results["SCGS"]["primal_gap"],
    results["CSFW"]["primal_gap"],
    results["SVRFW"]["primal_gap"],
]

x_limits = [
    min(
        np.min(results["SFW"]["time"]),
        np.min(results["SPIDER"]["time"]),
        np.min(results["SFW (momentum)"]["time"]),
        np.min(results["1-SFW"]["time"]),
        np.min(results["SCGS"]["time"]),
        np.min(results["SVRFW"]["time"]),
    ),
    None,
]

plot_results(
    list_x,
    list_data,
    [],
    "",
    r"$\mathrm{Time \ (s)}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=x_limits,
    save_figure=output_filepath,
    figure_width=figure_size,
)

