# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:05:51 2021

@author: pccom
"""
import numpy as np
import sys, os
import pickle

from frankwolfe.application_specific_algorithms import (
    frank_wolfe_optimal_experiment_design,
    away_frank_wolfe_optimal_experiment_design,
)
from frankwolfe.objective_functions import optimal_experiment_design
from frankwolfe.feasible_regions import probability_simplex

n = 100
p = 1000
num_steps = 100000

mu = np.zeros(n)
sigma = np.random.rand(n, n)
sigma = np.dot(sigma.T, sigma)
A = np.random.multivariate_normal(mu, sigma, size=p)
function = optimal_experiment_design(A)

LPOracle = probability_simplex(p)
# Define initial point in the interior of the simplex
initial_point = np.ones(p) / p

data_away_optimized_reference_solution = away_frank_wolfe_optimal_experiment_design(
    initial_point,
    function,
    LPOracle,
    algorithm_parameters={
        "maximum_time": 14400.0,
        "maximum_iterations": int(10 * num_steps),
        "stopping_frank_wolfe_gap": 1.0e-8,
    },
)

# Use vanilla FW with line search
data_line_seach_optimized_results = frank_wolfe_optimal_experiment_design(
    initial_point,
    function,
    LPOracle,
    algorithm_parameters={
        "maximum_time": 7200.0,
        "maximum_iterations": int(10 * num_steps),
    },
)

data_away_optimized_results = away_frank_wolfe_optimal_experiment_design(
    initial_point,
    function,
    LPOracle,
    algorithm_parameters={
        "maximum_time": 7200.0,
        "maximum_iterations": int(10 * num_steps),
    },
)


data_FW_optimized = {
    "name": r"FW Line search optimized",
    "f_value": data_line_seach_optimized_results["function_eval"],
    "primal_gap": [
        elem - data_away_optimized_reference_solution["function_eval"][-1]
        for elem in data_line_seach_optimized_results["function_eval"]
    ],
    "dual_gap": data_line_seach_optimized_results["frank_wolfe_gap"],
    "time": data_line_seach_optimized_results["timing"],
}
data_away_optimized = {
    "name": r"AFW Line search (optimized)",
    "f_value": data_away_optimized_results["function_eval"],
    "primal_gap": [
        elem - data_away_optimized_reference_solution["function_eval"][-1]
        for elem in data_away_optimized_results["function_eval"]
    ],
    "dual_gap": data_away_optimized_results["frank_wolfe_gap"],
    "time": data_away_optimized_results["timing"],
}


experiment_details = {
    "n": n,
    "p": p,
    "ref_solution": data_away_optimized_reference_solution["function_eval"][-1],
}
results = {
    "FW": data_FW_optimized,
    "AFW": data_away_optimized,
    "details": experiment_details,
}

output_directory = os.path.join(os.getcwd(), "output_data")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
    
output_directory_images = os.path.join(os.getcwd(), "output_images")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory_images):
    os.makedirs(output_directory_images)

# Output the results as a pickled object for later use.
filepath = os.path.join(os.getcwd(), "output_data", "optimal_experiment_design.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

from frankwolfe.plotting_function import plot_results

list_x_label = [
    np.arange(len(results["FW"]["primal_gap"])) + 1,
    np.arange(len(results["AFW"]["primal_gap"])) + 1,
]
list_data = [
    results["FW"]["primal_gap"],
    results["AFW"]["primal_gap"],
]
list_legend = [r"$\mathrm{FW}$", r"$\mathrm{AFW}$"]

colors = ["k", "c"]
markers = ["o", "s"]
colors_fill = ["None", "None"]

figure_size = 7.3

output_filepath = os.path.join(
    output_directory_images,
    "optimal_experiment_primal_gap_iteration.pdf",
)

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Iteration \ (t)}$",
    r"$- \ln \frac{\det V(x_t)}{\det V(x^*)} $",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=[
        1.0,
        1.0
        + max(
            len(results["FW"]["dual_gap"]),
            len(results["AFW"]["dual_gap"]),
        ),
    ],
    y_limits=[
        1.0e-11,
        None,
    ],
    legend_location="lower left",
    save_figure=output_filepath,
    # save_figure=None,
    figure_width=figure_size,
)


output_filepath = os.path.join(
    output_directory_images,
    "optimal_experiment_primal_gap_time.pdf",
)

list_x_label = [
    results["FW"]["time"],
    results["AFW"]["time"],
]

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Time \ (s)}$",
    r"$- \ln \frac{\det V(x_t)}{\det V(x^*)} $",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=[
        min(
            np.min(results["FW"]["time"]),
            np.min(results["AFW"]["time"]),
        ),
        max(
            np.max(results["FW"]["time"]),
            np.max(results["AFW"]["time"]),
        ),
    ],
    y_limits=[
        1.0e-11,
        None,
    ],
    legend_location="lower left",
    save_figure=output_filepath,
    # save_figure=None,
    figure_width=figure_size,
)