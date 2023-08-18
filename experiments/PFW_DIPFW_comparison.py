# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:37:39 2020

@author: pccom
"""
import sys, os
import numpy as np
import pickle


from frankwolfe.algorithms import away_frank_wolfe, decomposition_invariant_pairwise_frank_wolfe
from frankwolfe.feasible_regions import birkhoff_polytope
from frankwolfe.objective_functions import quadratic
from frankwolfe.auxiliary_functions import (
    step_size_class,
    stopping_criterion,
)

dimension = 3600
tol = 1.0e-9
numSteps = 1000

LPOracle = birkhoff_polytope(dimension)

mat = np.random.rand(dimension, dimension)
vector = np.random.rand(dimension)
matrix = mat.T.dot(mat)
function = quadratic(matrix, vector)
u, v = np.linalg.eig(matrix)

initial_point = LPOracle.initial_point()
# Compute optimal solution using NAGD
reference_results = decomposition_invariant_pairwise_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria = stopping_criterion({"timing": 600.0,"iterations": int(10.0 * numSteps), "frank_wolfe_gap":tol}),
)
fValOpt = reference_results["function_eval"][-1]

line_search = step_size_class("line_search")
stopping_criteria = stopping_criterion({"timing": 60,"iterations": numSteps, "frank_wolfe_gap":tol})

FW_pairwise = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    algorithm_parameters={
        "type_step": "pairwise",
    },
    stopping_criteria=stopping_criteria,
    step_size=line_search,
)

FW_DIPFW = decomposition_invariant_pairwise_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criteria,
)


# Put the results in a picle object and then output
paiwise_FW = {
    "name": "PFW",
    "primal_gap": [y - fValOpt for y in FW_pairwise["function_eval"]],
    "timing": FW_pairwise["timing"],
}
DI_FW = {
    "name": "DIPFW",
    "primal_gap": [y - fValOpt for y in FW_DIPFW["function_eval"]],
    "timing": FW_DIPFW["timing"],
}

eigenvalues = {
    "name": "eigenvalues",
    "values": u,
}

results = {
    "optimum": reference_results["function_eval"][-1],
    "paiwise_FW": paiwise_FW,
    "DI_FW": DI_FW,
    "eigenvalues": eigenvalues,
}

output_directory = os.path.join(os.getcwd(), "output_data")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

# Output the results as a pickled object for later use.
filepath = os.path.join(os.getcwd(), "output_data", "PFW_DIPFW_comparison.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

output_directory_images = os.path.join(os.getcwd(), "output_images")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory_images):
    os.makedirs(output_directory_images)

from frankwolfe.plotting_function import plot_results

output_filepath = os.path.join(
    output_directory_images,
    "PFW_DIPFW_primal_gap_iteration.pdf",
)

list_x_label = [
    np.arange(len(results["paiwise_FW"]["primal_gap"])) + 1,
    np.arange(len(results["DI_FW"]["primal_gap"])) + 1,
]
list_data = [results["paiwise_FW"]["primal_gap"], results["DI_FW"]["primal_gap"]]
list_legend = [r"$\mathrm{PFW}$", r"$\mathrm{DI-PFW}$"]


colors = ["b", "g"]
markers = ["^", "D"]
colors_fill = ["None", "None"]

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
        + max(len(results["paiwise_FW"]["primal_gap"]), len(results["DI_FW"]["primal_gap"])),
    ],
    y_limits=[
        1.0e-8,
        max(
            np.max(results["paiwise_FW"]["primal_gap"]),
            np.max(results["DI_FW"]["primal_gap"]),
        ),
    ],
    save_figure=output_filepath,
    figure_width=figure_size,
)


list_x_label = [results["paiwise_FW"]["timing"], results["DI_FW"]["timing"]]

output_filepath = os.path.join(
    output_directory_images,
    "PFW_DIPFW_primal_gap_time.pdf",
)

plot_results(
    list_x_label,
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
    x_limits=[
        min(
            np.min(results["paiwise_FW"]["timing"]),
            np.min(results["DI_FW"]["timing"]),
        ),
        max(
            np.max(results["paiwise_FW"]["timing"]),
            np.max(results["DI_FW"]["timing"]),
        ),
    ],
    y_limits=[
        1.0e-8,
        max(
            np.max(results["paiwise_FW"]["primal_gap"]),
            np.max(results["DI_FW"]["primal_gap"]),
        ),
    ],
    save_figure=output_filepath,
    figure_width=figure_size,
)
