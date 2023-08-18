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
    conditional_gradient_sliding,
)
from frankwolfe.feasible_regions import (
    nuclear_norm_ball,
)
from frankwolfe.objective_functions import quadratic
from frankwolfe.auxiliary_functions import rvs, step_size_class, stopping_criterion

# -----------------------L1 Ball example (cvx).------------------------------

dimension = 100
tol = 1.0e-9
numSteps = 10000

# LPOracle = spectrahedron(dimension)
LPOracle = nuclear_norm_ball(10, 10, alpha=1.0, flatten=True)

mat = np.random.rand(dimension, dimension)
vector = np.random.rand(dimension)
matrix = mat.T.dot(mat)
function = quadratic(matrix, vector)
u, v = np.linalg.eig(matrix)

# Build the convex function with a Hessian that has exactly one zero.
H = rvs(dimension)
u = np.sort(u)
u[0] = 0
matrix = np.zeros((dimension, dimension))
for i in range(dimension):
    matrix += u[i] * np.outer(H[:, i], H[:, i])

function = quadratic(matrix, vector)

initial_point = LPOracle.initial_point()

opt = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": 100000, "frank_wolfe_gap":tol}),
    algorithm_parameters={
        "algorithm_type": "lazy",
    },
    step_size =step_size_class("line_search"),
)
fValOpt = function.f(opt["solution"])

CGS = conditional_gradient_sliding(
    function,
    LPOracle,
    x0 = initial_point,
    algorithm_parameters = {"monitor_frank_wolfe_gap": False},
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
)


FW_vanilla_constant_stepsize = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": CGS["LMO_calls"][-1]}),
    step_size =step_size_class("constant_step"),
)


FW_vanilla_short_step = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": CGS["LMO_calls"][-1]}),
    step_size =step_size_class("short_step"),
)

line_search = {"type_step": "line_search"}

FW_vanilla_line_search = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": CGS["LMO_calls"][-1]}),
    step_size =step_size_class("line_search"),
)


# Put the results in a picle object and then output
vanilla_FW_constant_step = {
    "name": "vanilla_FW_constant_step",
    "primal_gap": [y - fValOpt for y in FW_vanilla_constant_stepsize["function_eval"]],
    "frank_wolfe_gap": FW_vanilla_constant_stepsize["frank_wolfe_gap"],
    "timing": FW_vanilla_constant_stepsize["timing"],
}

vanilla_FW_short_step = {
    "name": "vanilla_FW_short_step",
    "primal_gap": [y - fValOpt for y in FW_vanilla_short_step["function_eval"]],
    "frank_wolfe_gap": FW_vanilla_short_step["frank_wolfe_gap"],
    "timing": FW_vanilla_short_step["timing"],
}

vanilla_FW_line_search = {
    "name": "vanilla_FW_line_search",
    "primal_gap": [y - fValOpt for y in FW_vanilla_line_search["function_eval"]],
    "frank_wolfe_gap": FW_vanilla_line_search["frank_wolfe_gap"],
    "timing": FW_vanilla_line_search["timing"],
}

CGS_data = {
    "name": "CGS",
    "primal_gap": [y - fValOpt for y in CGS["function_eval"]],
    "frank_wolfe_gap": CGS["frank_wolfe_gap"],
    "FOO_calls": CGS["FOO_calls"],
    "LMO_calls": CGS["LMO_calls"],
    "timing": CGS["timing"],
}

results = {
    "optimum": opt,
    "L": np.max(u),
    "mu": 0.0,
    "vanilla_FW_constant_step": vanilla_FW_constant_step,
    "vanilla_FW_line_search": vanilla_FW_line_search,
    "vanilla_FW_short_step": vanilla_FW_short_step,
    "CGS": CGS_data,
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
filepath = os.path.join(os.getcwd(), "output_data", "FW_CGS_comparison_cvx.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

output_filepath = os.path.join(
    output_directory_images,
    "FW_CGS_primal_gap_LMO_cvx.pdf",
)

max_number_CGS = 3300

values = np.zeros(len(results["vanilla_FW_constant_step"]["primal_gap"]))
for i in range(len(values)):
    values[i] = 30 * results["L"] / (2 * (i + 1) * (i + 2))

figure_size = 7.3

list_x_label = [
    np.arange(len(results["vanilla_FW_constant_step"]["primal_gap"])) + 1,
    np.arange(len(results["vanilla_FW_short_step"]["primal_gap"])) + 1,
    np.arange(len(results["vanilla_FW_line_search"]["primal_gap"])) + 1,
    np.asarray(results["CGS"]["LMO_calls"][:max_number_CGS]) + 1,
]
list_data = [
    results["vanilla_FW_constant_step"]["primal_gap"],
    results["vanilla_FW_short_step"]["primal_gap"],
    results["vanilla_FW_line_search"]["primal_gap"],
    results["CGS"]["primal_gap"][:max_number_CGS],
]

from frankwolfe.plotting_function import plot_results

colors = ["k", "b", "r", "c", "--m"]
markers = ["o", "s", "^", "P", None]
colors_fill = ["None", "None", "None", "None", "None"]

list_legend = [
    r"$\mathrm{FW \ (line \ search)}$",
    r"$\mathrm{FW \ (short \ step)}$",
    r"$\mathrm{FW \ } (2/(t + 2)$)",
    r"$\mathrm{CGS}$",
    r"$\mathrm{CGS \ FOO \ bound}$",
]

plot_results(
    list_x_label,
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
    x_limits=[
        1.0,
        1.0
        + max(
            len(results["vanilla_FW_constant_step"]["primal_gap"]),
            len(results["vanilla_FW_short_step"]["primal_gap"]),
            len(results["vanilla_FW_line_search"]["primal_gap"]),
            results["CGS"]["LMO_calls"][:max_number_CGS][-1],
        ),
    ],
    save_figure=output_filepath,
    # save_figure=None,
    legend_location="lower left",
    figure_width=figure_size,
)

output_filepath = os.path.join(
    output_directory_images,
    "FW_CGS_primal_gap_FOO_cvx.pdf",
)

list_x_label = [
    np.arange(len(results["vanilla_FW_constant_step"]["primal_gap"])) + 1,
    np.arange(len(results["vanilla_FW_short_step"]["primal_gap"])) + 1,
    np.arange(len(results["vanilla_FW_line_search"]["primal_gap"])) + 1,
    np.asarray(results["CGS"]["FOO_calls"][:max_number_CGS]) + 1,
    np.arange(len(results["vanilla_FW_constant_step"]["primal_gap"])) + 1,
]
list_data = [
    results["vanilla_FW_constant_step"]["primal_gap"],
    results["vanilla_FW_short_step"]["primal_gap"],
    results["vanilla_FW_line_search"]["primal_gap"],
    results["CGS"]["primal_gap"][:max_number_CGS],
    values,
]

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Number \ of \ FOO \ calls}$",
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
            len(results["vanilla_FW_constant_step"]["primal_gap"]),
            len(results["vanilla_FW_short_step"]["primal_gap"]),
            len(results["vanilla_FW_line_search"]["primal_gap"]),
            results["CGS"]["FOO_calls"][:max_number_CGS][-1],
        ),
    ],
    save_figure=output_filepath,
    # save_figure=None,
    legend_location="lower left",
    figure_width=figure_size,
)
