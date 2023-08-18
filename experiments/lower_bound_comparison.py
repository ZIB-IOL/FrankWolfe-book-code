# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:37:39 2020

@author: pccom
"""
import sys, os
import numpy as np
import pickle

from frankwolfe.algorithms import frank_wolfe
from frankwolfe.feasible_regions import probability_simplex
from frankwolfe.objective_functions import quadratic
from frankwolfe.auxiliary_functions import step_size_class, stopping_criterion

dimension = 1000
numSteps = 100000

feasible_region = probability_simplex(dimension)
function = quadratic(2.0 * np.identity(dimension), np.zeros(dimension))
initial_point = feasible_region.initial_point()
# Compute optimal solution using NAGD
opt = np.ones(dimension) / dimension
fValOpt = function.f(opt)

# Run the standard stepsize.
data = frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("constant_step"),
)

# Values for the lower bound
lower_bound = np.zeros(dimension)
for i in range(dimension):
    lower_bound[i] = 1.0 / (i + 1.0) - 1.0 / dimension

# Put the results in a picle object and then output
standard_results = {
    "name": r"$2/(t + 2)$",
    "f_value": data["function_eval"],
    "primal_gap": [y - fValOpt for y in data["function_eval"]],
    "dual_gap": data["frank_wolfe_gap"],
    "time": data["timing"],
}

lower_bound = {
    "name": "Lower bound",
    "primal_gap": lower_bound,
}

experiment_details = {
    "dimension": dimension,
    "numSteps": numSteps,
    "feasible_region": "L1_ball",
}
results = {
    "details": experiment_details,
    "standard": standard_results,
    "lower_bound": lower_bound,
}

output_directory = os.path.join(os.getcwd(), "output_data")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
    
# Output the results as a pickled object for later use.
filepath = os.path.join(os.getcwd(), "output_data", "lower_bound_results.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)
    
    
output_directory_images = os.path.join(os.getcwd(), "output_images")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory_images):
    os.makedirs(output_directory_images)
    
output_filepath = os.path.join(
    output_directory_images,
    "lower_bound_comparison.pdf",
)

list_x_label = [
    np.arange(len(results["standard"]["primal_gap"])),
    np.arange(len(results["lower_bound"]["primal_gap"])),
]
list_data = [results["standard"]["primal_gap"], results["lower_bound"]["primal_gap"]]
list_legend = [
    r"$\mathrm{FW \ }(\gamma_t = 2/(t+2)$)",
    r"$\mathrm{Lower \ bound}$",
]

colors = ["k", "b"]
markers = ["o", "s"]
colors_fill = ["None", "None"]

figure_size = 8
from frankwolfe.plotting_function import plot_results

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Number \ of \ LMO \ calls}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    save_figure=output_filepath,
    figure_width=figure_size,
)
