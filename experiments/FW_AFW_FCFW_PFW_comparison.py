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
    fully_corrective_frank_wolfe,
    accelerated_projected_gradient_descent,
)
from frankwolfe.auxiliary_functions import (
    step_size_class,
)
from frankwolfe.feasible_regions import probability_simplex
from frankwolfe.objective_functions import quadratic


# -----------------------L1 Ball example.------------------------------

dimension = 100
tol = 1.0e-4
numSteps = 10000

LPOracle = probability_simplex(dimension)

mat = np.random.rand(dimension, dimension)
vector = np.random.rand(dimension)
matrix = mat.T.dot(mat)
function = quadratic(matrix, vector)
u, v = np.linalg.eig(matrix)

initial_point = LPOracle.initial_point()
# Compute optimal solution using NAGD
opt = accelerated_projected_gradient_descent(initial_point, function, LPOracle, tol)
fValOpt = function.f(opt)
line_search = step_size_class("line_search")

frank_wolfe_data = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size=line_search,
    algorithm_parameters={
        "algorithm_type": "standard",
    },
)

away_frank_wolfe_data = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size= line_search,
    algorithm_parameters={
        "algorithm_type": "standard",
    },
)

pairwise_frank_wolfe_data = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size= line_search,
    algorithm_parameters={
        "algorithm_type": "pairwise",
    },
)

fully_corrective_frank_wolfe_data = fully_corrective_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    algorithm_parameters={"inner_stopping_frank_wolfe_gap": 1.0e-8},
    step_size = line_search,
)

# Put the results in a picle object and then output
frank_wolfe_results = {
    "name": "vanilla_FW",
    "primal_gap": [y - fValOpt for y in frank_wolfe_data["function_eval"]],
    "time": frank_wolfe_data["timing"],
}
away_frank_wolfe_results = {
    "name": "AFW",
    "primal_gap": [y - fValOpt for y in away_frank_wolfe_data["function_eval"]],
    "time": away_frank_wolfe_data["timing"],
}
pairwise_frank_wolfe_results = {
    "name": "PFW",
    "primal_gap": [y - fValOpt for y in pairwise_frank_wolfe_data["function_eval"]],
    "time": pairwise_frank_wolfe_data["timing"],
}
fully_corrective_frank_wolfe_results = {
    "name": "FCFW",
    "primal_gap": [
        y - fValOpt for y in fully_corrective_frank_wolfe_data["function_eval"]
    ],
    "time": fully_corrective_frank_wolfe_data["timing"],
}

eigenvalues = {
    "name": "eigenvalues",
    "values": u,
}

results = {
    "optimum": opt,
    "vanilla_FW": frank_wolfe_results,
    "away_FW": away_frank_wolfe_results,
    "paiwise_FW": pairwise_frank_wolfe_results,
    "fully_corrective_FW": fully_corrective_frank_wolfe_results,
    "eigenvalues": eigenvalues,
}


output_directory = os.path.join(os.getcwd(), "output_data")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
    
# Output the results as a pickled object for later use.
filepath = os.path.join(os.getcwd(), "output_data", "FW_AFW_FCFW_PFW_comparison.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

output_directory_images = os.path.join(os.getcwd(), "output_images")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory_images):
    os.makedirs(output_directory_images)

from frankwolfe.plotting_function import plot_results

figure_size = 7.3

output_filepath = os.path.join(
    output_directory_images,
    "FW_AFW_FCFW_PFW_primal_gap.pdf",
)

list_x_label = [
    np.arange(len(results["vanilla_FW"]["primal_gap"])) + 1,
    np.arange(len(results["away_FW"]["primal_gap"])) + 1,
    np.arange(len(results["paiwise_FW"]["primal_gap"])) + 1,
    np.arange(len(results["fully_corrective_FW"]["primal_gap"])) + 1,
]
list_data = [
    results["vanilla_FW"]["primal_gap"],
    results["away_FW"]["primal_gap"],
    results["paiwise_FW"]["primal_gap"],
    results["fully_corrective_FW"]["primal_gap"],
]
list_legend = [
    r"$\mathrm{FW}$",
    r"$\mathrm{AFW}$",
    r"$\mathrm{PFW}$",
    r"$\mathrm{FCFW}$",
]

colors = ["k", "c", "b", "m"]
markers = ["o", "s", "^", "P"]
colors_fill = ["None", "None", "None", "None"]

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
            len(results["away_FW"]["primal_gap"]),
            len(results["paiwise_FW"]["primal_gap"]),
            len(results["fully_corrective_FW"]["primal_gap"]),
        ),
    ],
    save_figure=output_filepath,
    figure_width=figure_size,
)

list_x_label = [
    results["vanilla_FW"]["time"],
    results["away_FW"]["time"],
    results["paiwise_FW"]["time"],
    results["fully_corrective_FW"]["time"],
]

output_filepath = os.path.join(
    output_directory_images,
    "FW_AFW_FCFW_PFW_primal_gap_time.pdf",
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
            np.min(results["vanilla_FW"]["time"]),
            np.min(results["away_FW"]["time"]),
            np.min(results["paiwise_FW"]["time"]),
            np.min(results["fully_corrective_FW"]["time"]),
        ),
        max(
            np.max(results["vanilla_FW"]["time"]),
            np.max(results["away_FW"]["time"]),
            np.max(results["paiwise_FW"]["time"]),
            np.max(results["fully_corrective_FW"]["time"]),
        ),
    ],
    save_figure=output_filepath,
    figure_width=figure_size,
)
