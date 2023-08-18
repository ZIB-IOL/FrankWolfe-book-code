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
    away_frank_wolfe,
    projected_gradient_descent,
)
from frankwolfe.feasible_regions import L1_ball
from frankwolfe.objective_functions import  quadratic
from frankwolfe.auxiliary_functions import step_size_class, stopping_criterion


# -----------------------L1 Ball example.------------------------------

dimension = 200
tol = 1.0e-9
numSteps = 10000

LPOracle = L1_ball(dimension)

mat = np.random.rand(dimension, dimension)
vector = np.random.rand(dimension)
matrix = mat.T.dot(mat)
function = quadratic(matrix, vector)
u, v = np.linalg.eig(matrix)

initial_point = LPOracle.initial_point()
# Compute optimal solution using NAGD
opt = projected_gradient_descent(initial_point, function, LPOracle, tol)

fValOpt = function.f(opt)

# Run the standard stepsize.
FW_ada_away_results = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("adaptive_short_step"),
)

# Run the short-step rule stepsize.
FW_away_results = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("short_step"),
)

# Run the line search stepsize.
FW_ada_pairwise_results = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={"algorithm_type": "pairwise"},
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("adaptive_short_step"),
)


# Run the line search stepsize.
FW_pairwise_results = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={"algorithm_type": "pairwise"},
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("short_step"),
)


# Run the line search stepsize.
FW_ada_vanilla_results = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("adaptive_short_step"),
)

# Run the line search stepsize.
FW_vanilla_results = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("short_step"),
)

# Put the results in a picle object and then output
vanilla_FW = {
    "name": "vanilla_FW",
    "f_value": FW_vanilla_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_vanilla_results["function_eval"]],
    "dual_gap": FW_vanilla_results["frank_wolfe_gap"],
    "time": FW_vanilla_results["timing"],
}
ada_vanilla_FW = {
    "name": "ada_vanilla_FW",
    "smoothness_estimate": FW_ada_vanilla_results["L_estimate"],
    "f_value": FW_ada_vanilla_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_ada_vanilla_results["function_eval"]],
    "dual_gap": FW_ada_vanilla_results["frank_wolfe_gap"],
    "time": FW_ada_vanilla_results["timing"],
}
away_FW = {
    "name": "AFW",
    "f_value": FW_away_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_away_results["function_eval"]],
    "dual_gap": FW_away_results["frank_wolfe_gap"],
    "time": FW_away_results["timing"],
}

ada_away_FW = {
    "name": "ada_AFW",
    "smoothness_estimate": FW_ada_away_results["L_estimate"],
    "f_value": FW_ada_away_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_ada_away_results["function_eval"]],
    "dual_gap": FW_ada_away_results["frank_wolfe_gap"],
    "time": FW_ada_away_results["timing"],
}

paiwise_FW = {
    "name": "PFW",
    "f_value": FW_pairwise_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_pairwise_results["function_eval"]],
    "dual_gap": FW_pairwise_results["frank_wolfe_gap"],
    "time": FW_pairwise_results["timing"],
}

ada_pairwise_FW = {
    "name": "ada_PFW",
    "smoothness_estimate": FW_ada_pairwise_results["L_estimate"],
    "f_value": FW_ada_pairwise_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_ada_pairwise_results["function_eval"]],
    "dual_gap": FW_ada_pairwise_results["frank_wolfe_gap"],
    "time": FW_ada_pairwise_results["timing"],
}

eigenvalues = {
    "name": "eigenvalues",
    "values": u,
}

results = {
    "vanilla_FW": vanilla_FW,
    "ada_vanilla_FW": ada_vanilla_FW,
    "away_FW": away_FW,
    "ada_away_FW": ada_away_FW,
    "paiwise_FW": paiwise_FW,
    "ada_pairwise_FW": ada_pairwise_FW,
    "eigenvalues": eigenvalues,
}

# Output the results as a pickled object for later use.
filepath = os.path.join(os.getcwd(), "output_data", "adaptive_comparison.pickle")

output_directory = os.path.join(os.getcwd(), "output_data")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
    
output_directory_images = os.path.join(os.getcwd(), "output_images")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory_images):
    os.makedirs(output_directory_images)

with open(filepath, "wb") as f:
    pickle.dump(results, f)

from frankwolfe.plotting_function import plot_results

figure_size = 7.3

list_x_label = [
    np.arange(len(results["vanilla_FW"]["primal_gap"])),
    np.arange(len(results["ada_vanilla_FW"]["primal_gap"])),
    np.arange(len(results["paiwise_FW"]["primal_gap"])),
    np.arange(len(results["ada_pairwise_FW"]["primal_gap"])),
]
list_data = [
    results["vanilla_FW"]["primal_gap"],
    results["ada_vanilla_FW"]["primal_gap"],
    results["paiwise_FW"]["primal_gap"],
    results["ada_pairwise_FW"]["primal_gap"],
]
list_legend = [
    r"$\mathrm{FW}$",
    r"$\mathrm{AdaFW}$",
    r"$\mathrm{PFW}$",
    r"$\mathrm{AdaPFW}$",
]

output_filepath = os.path.join(
    output_directory_images,
    "adaptive_stepsize_primal_gap.pdf",
)

colors = ["k", "c", "r", "g"]
markers = ["X", "s", "D", "p"]
colorfills = ["none", "none", "none", "none"]

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Iteration \ (t)}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colorfills,
    log_x=True,
    log_y=True,
    save_figure=output_filepath,
    figure_width=figure_size,
)

list_x_label = [
    np.arange(len(results["ada_vanilla_FW"]["smoothness_estimate"])),
    np.arange(len(results["ada_pairwise_FW"]["smoothness_estimate"])),
]
list_data = [
    results["ada_vanilla_FW"]["smoothness_estimate"],
    results["ada_pairwise_FW"]["smoothness_estimate"],
]
list_legend = ["AdaFW", "AdaPFW"]
colors = ["c", "g"]
markers = ["s", "p"]

output_filepath = os.path.join(
    output_directory_images,
    "adaptive_stepsize_smoothness_estimate.pdf",
)

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Iteration \ (t)}$",
    r"$\mathrm{Smoothness \ estimate}$",
    colors,
    markers,
    colorfills=colorfills,
    y_limits=[10, 1000],
    log_x=True,
    log_y=True,
    save_figure=output_filepath,
    figure_width=figure_size,
)