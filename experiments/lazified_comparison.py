# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:37:39 2020

@author: pccom
"""
import sys, os
import numpy as np
import pickle
import time, datetime

from frankwolfe.algorithms import (
    away_frank_wolfe,
    frank_wolfe,
    accelerated_projected_gradient_descent,
)
from frankwolfe.feasible_regions import nuclear_norm_ball, birkhoff_polytope
from frankwolfe.objective_functions import quadratic_type_3, quadratic

from frankwolfe.auxiliary_functions import (
    step_size_class,
    stopping_criterion,
)

# -----------------------L1 Ball example.------------------------------

dimension = 900
tol = 1.0e-3
numSteps = 10000
# numSteps = 50000
maximum_time = 600

# LPOracle = spectrahedron(dimension)
# LPOracle = nuclear_norm_ball(30,30, alpha=1.0, flatten = True)
LPOracle = birkhoff_polytope(dimension)

mat = np.random.rand(dimension, dimension)
vector = np.random.rand(dimension)
matrix = mat.T.dot(mat)
function = quadratic(matrix, vector)
u, v = np.linalg.eig(matrix)

initial_point = LPOracle.initial_point()

# Run the short-step rule stepsize.
reference_results = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("line_search"),
    algorithm_parameters={
        "algorithm_type": "lazy",
        "maximum_time": maximum_time,
        "maximum_iterations": int(10.0 * numSteps),
    },
)
fValOpt = reference_results["function_eval"][-1]

# Run the short-step rule stepsize.
FW_away_lazy_results = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("short_step"),
    algorithm_parameters={
        "algorithm_type": "lazy",
        "maximum_time": maximum_time,
        "maximum_iterations": numSteps,
    },
)
# Run the standard stepsize.
FW_away_results = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("short_step"),
    algorithm_parameters={
        "algorithm_type": "standard",
        "maximum_time": maximum_time,
        "maximum_iterations": numSteps,
    },
)

# Run the short-step rule stepsize.
FW_results = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("short_step"),
    algorithm_parameters={
        "algorithm_type": "standard",
        "maximum_time": maximum_time,
        "maximum_iterations": numSteps,
        "maintain_active_set": True,
    },
)
# Run the standard stepsize.
FW_lazy_results = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("short_step"),
    algorithm_parameters={
        "algorithm_type": "lazy",
        "maximum_time": maximum_time,
        "maximum_iterations": numSteps,
    },
)

away_FW = {
    "name": "AFW",
    "f_value": FW_away_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_away_results["function_eval"]],
    "dual_gap": FW_away_results["frank_wolfe_gap"],
    "time": FW_away_results["timing"],
}

lazy_away_FW = {
    "name": "lazy_AFW",
    "f_value": FW_away_lazy_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_away_lazy_results["function_eval"]],
    "dual_gap": FW_away_lazy_results["frank_wolfe_gap"],
    "time": FW_away_lazy_results["timing"],
}

FW = {
    "name": "FW",
    "f_value": FW_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_results["function_eval"]],
    "dual_gap": FW_results["frank_wolfe_gap"],
    "time": FW_results["timing"],
}

lazy_FW = {
    "name": "lazy_FW",
    "f_value": FW_lazy_results["function_eval"],
    "primal_gap": [y - fValOpt for y in FW_lazy_results["function_eval"]],
    "dual_gap": FW_lazy_results["frank_wolfe_gap"],
    "time": FW_lazy_results["timing"],
}

eigenvalues = {
    "name": "eigenvalues",
    "values": u,
}

results = {
    "away_FW": away_FW,
    "lazy_away_FW": lazy_away_FW,
    "FW": FW,
    "lazy_FW": lazy_FW,
    "eigenvalues": eigenvalues,
}


ts = time.time()
timestamp = (
    datetime.datetime.fromtimestamp(ts)
    .strftime("%Y-%m-%d %H:%M:%S")
    .replace(" ", "-")
    .replace(":", "-")
)

output_directory = os.path.join(os.getcwd(), "output_data")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
    
output_directory_images = os.path.join(os.getcwd(), "output_images")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory_images):
    os.makedirs(output_directory_images)

# Output the results as a pickled object for later use.
filepath = os.path.join(
    os.getcwd(), "output_data", "lazified_version_" + str(timestamp) + ".pickle"
)
with open(filepath, "wb") as f:
    pickle.dump(results, f)

figure_size = 7.3

from frankwolfe.plotting_function import plot_results

output_filepath = os.path.join(
    output_directory_images,
    "lazification_primal_gap_iteration.pdf",
)

list_x_label = [
    np.arange(len(results["FW"]["primal_gap"])),
    np.arange(len(results["away_FW"]["primal_gap"])),
    np.arange(len(results["lazy_FW"]["primal_gap"])),
    np.arange(len(results["lazy_away_FW"]["primal_gap"])),
]
list_data = [
    results["FW"]["primal_gap"],
    results["away_FW"]["primal_gap"],
    results["lazy_FW"]["primal_gap"],
    results["lazy_away_FW"]["primal_gap"],
]
list_legend = [
    r"$\mathrm{FW}$",
    r"$\mathrm{AFW}$",
    r"$\mathrm{LCG}$",
    r"$\mathrm{Lazy \ AFW}$",
]

colors = ["b", "r", "k", "m"]
markers = ["o", "s", "^", "P"]
colors_fill = ["None", "None", "None", "None"]


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
    save_figure=output_filepath,
    figure_width=figure_size,
)

list_x_label = [
    results["FW"]["time"],
    results["away_FW"]["time"],
    results["lazy_FW"]["time"],
    results["lazy_away_FW"]["time"],
]
list_legend = []

output_filepath = os.path.join(
    output_directory_images,
    "lazification_primal_gap_time.pdf",
)

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Time \ (s)}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    save_figure=output_filepath,
    figure_width=figure_size,
)
