# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:37:39 2020

@author: pccom
"""
import sys, os
import numpy as np
import pickle

from frankwolfe.algorithms import (
    frank_wolfe,
)
from frankwolfe.feasible_regions import probability_simplex, box_constraints
from frankwolfe.objective_functions import quadratic_sparse_signal_recovery
from frankwolfe.auxiliary_functions import step_size_class, stopping_criterion


# # # -----------------------Hypercube example.------------------------------

tol = 1.0e-9
numSteps = 10000

dimension = 1200
number_of_samples = 600
A = np.random.normal(loc=0.0, scale=1.0, size=(number_of_samples, dimension))
opt = np.random.choice([0, 1], size=(dimension,), p=[0.5, 0.5])
opt[:5] = 0.5
b = A.dot(opt)
function = quadratic_sparse_signal_recovery(A, b)

LPOracle = box_constraints(np.zeros(dimension), np.ones(dimension))

# initial_point = LPOracle.initial_point()
initial_point = opt.copy().astype(float)
initial_point[:5] = 0.0
if opt[6] == 1:
    initial_point[6] = 0.0
else:
    initial_point[6] = 1.0
# Compute optimal solution using NAGD

fValOpt = 0.0

FW_vanilla_results = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size = step_size_class("constant_step"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap": 1.0e-3}),
)

FW_NEP_results = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "nep",
    },
    x0 = initial_point,
    step_size = step_size_class("constant_step"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap": 1.0e-3}),

)

# Put the results in a picle object and then output
vanilla_FW = {
    "name": "vanilla_FW",
    "primal_gap": [y - fValOpt for y in FW_vanilla_results["function_eval"]],
    "dual_gap": FW_vanilla_results["frank_wolfe_gap"],
}
NEP_FW = {
    "name": "NEP_FW",
    "primal_gap": [y - fValOpt for y in FW_NEP_results["function_eval"]],
    "dual_gap": FW_NEP_results["frank_wolfe_gap"],
}
results = {
    "optimum": opt,
    "vanilla_FW": vanilla_FW,
    "NEP_FW": NEP_FW,
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
filepath = os.path.join(os.getcwd(), "output_data", "FW_NEP_comparison_hypercube.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

from frankwolfe.plotting_function import plot_results

output_filepath = os.path.join(
    output_directory_images,
    "FW_NEP_hypercube_primal_gap.pdf",
)

list_x_label = [
    np.arange(len(results["NEP_FW"]["primal_gap"])) + 1,
    np.arange(len(results["vanilla_FW"]["primal_gap"])) + 1,
]
list_data = [
    results["NEP_FW"]["primal_gap"],
    results["vanilla_FW"]["primal_gap"],
]
list_legend = [
    r"$\mathrm{NEP-FW}$",
    r"$\mathrm{FW}$",
]
colors_fill = ["None", "None"]
colors = ["c", "k"]
markers = ["s", "o"]

figure_size = 7.3

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Iteration \ (t)}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=[
        1.0,
        1.0
        + max(
            len(results["vanilla_FW"]["primal_gap"]),
            len(results["NEP_FW"]["primal_gap"]),
        ),
    ],
    save_figure=output_filepath,
    figure_width=figure_size,
)


# # -----------------------Simplex example.------------------------------

opt = np.zeros(dimension)
opt[0:5] = np.random.rand(5)
opt /= np.sum(opt)
b = A.dot(opt)
function = quadratic_sparse_signal_recovery(A, b)

LPOracle = probability_simplex(dimension, alpha=1)
initial_point = LPOracle.initial_point()

fValOpt = 0.0

line_search = {
    "type_step": "constant_step",
    "additive_constant": 2,
    "multiplicative_constant": 2,
    "alpha_max": 1.0,
}

FW_vanilla_results = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size = step_size_class("constant_step"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap": 1.0e-3}),
)

FW_NEP_results = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "nep",
    },
    x0 = initial_point,
    step_size = step_size_class("constant_step"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap": 1.0e-3}),
)

# Put the results in a picle object and then output
vanilla_FW = {
    "name": "vanilla_FW",
    "primal_gap": [y - fValOpt for y in FW_vanilla_results["function_eval"]],
    "dual_gap": FW_vanilla_results["frank_wolfe_gap"],
}
NEP_FW = {
    "name": "NEP_FW",
    "primal_gap": [y - fValOpt for y in FW_NEP_results["function_eval"]],
    "dual_gap": FW_NEP_results["frank_wolfe_gap"],
}

results = {
    "optimum": opt,
    "vanilla_FW": vanilla_FW,
    "NEP_FW": NEP_FW,
}

# Output the results as a pickled object for later use.
filepath = os.path.join(os.getcwd(), "output_data", "FW_NEP_comparison_simplex.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

output_filepath = os.path.join(
    output_directory_images,
    "FW_NEP_simplex_primal_gap.pdf",
)

list_x_label = [
    np.arange(len(results["NEP_FW"]["primal_gap"])) + 1,
    np.arange(len(results["vanilla_FW"]["primal_gap"])) + 1,
]
list_data = [
    results["NEP_FW"]["primal_gap"],
    results["vanilla_FW"]["primal_gap"],
]
list_legend = [
    r"$\mathrm{NEP-FW}$",
    r"$\mathrm{FW}$",
]

colors = ["c", "k"]
markers = ["s", "o"]
colors_fill = ["None", "None"]

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Iteration \ (t)}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=[
        1.0,
        1.0
        + max(
            len(results["vanilla_FW"]["primal_gap"]),
            len(results["NEP_FW"]["primal_gap"]),
        ),
    ],
    save_figure=output_filepath,
    figure_width=figure_size,
)
