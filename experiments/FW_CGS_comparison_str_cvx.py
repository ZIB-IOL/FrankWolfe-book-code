# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:37:39 2020

@author: pccom
"""
import sys, os
import numpy as np
import pickle

from frankwolfe.algorithms import frank_wolfe, projected_gradient_descent, away_frank_wolfe
from frankwolfe.algorithms import conditional_gradient_sliding
from frankwolfe.feasible_regions import L1_ball
from frankwolfe.objective_functions import quadratic_type_3
from frankwolfe.auxiliary_functions import (
    step_size_class,
    stopping_criterion,
)


# -----------------------L1 Ball example.------------------------------

dimension = 500
tol = 1.0e-9
numSteps = 10000
# numSteps = 50000
Mu = 1.0
L = 100.0

LPOracle = L1_ball(dimension)
xOptGlobal = np.random.uniform(-1.0, 1.0, dimension)
xOptGlobal /= 5.0 * np.sum(np.abs(xOptGlobal))
function = quadratic_type_3(dimension, xOptGlobal, Mu=Mu, L=L)

initial_point = LPOracle.initial_point()
# Compute optimal solution using NAGD
opt = projected_gradient_descent(initial_point, function, LPOracle, tol)

fValOpt = function.f(opt)


CGS = conditional_gradient_sliding(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"timing": 60.0, "iterations": numSteps}),
)

Initial_FW_gap = np.dot(
    function.grad(initial_point),
    initial_point - LPOracle.linear_optimization_oracle(function.grad(initial_point)),
)

# Run the standard stepsize.
data_standard_step_size = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("constant_step"),
    stopping_criteria=stopping_criterion({"iterations": CGS["LMO_calls"][-1]}),
)

# Run the short-step rule stepsize.
data_short_step = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("short_step"),
    stopping_criteria=stopping_criterion({"iterations": CGS["LMO_calls"][-1]}),
)

# Run the line search stepsize.
data_line_seach = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": CGS["LMO_calls"][-1]}),
)

# Run the short-step rule stepsize.
data_away = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("short_step"),
    stopping_criteria=stopping_criterion({"timing": 7200.0,"iterations": CGS["LMO_calls"][-1]}),
)

# Run the short-step rule stepsize.
data_away_ls = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    step_size =step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 7200.0,"iterations": CGS["LMO_calls"][-1]}),
)

# Put the results in a picle object and then output
standard_results = {
    "name": r"$2/(t + 2)$",
    "f_value": data_standard_step_size["function_eval"],
    "primal_gap": [y - fValOpt for y in data_standard_step_size["function_eval"]],
    "dual_gap": data_standard_step_size["frank_wolfe_gap"],
    "time": data_standard_step_size["timing"],
}
short_step_results = {
    "name": r"Short step",
    "f_value": data_short_step["function_eval"],
    "primal_gap": [y - fValOpt for y in data_short_step["function_eval"]],
    "dual_gap": data_short_step["frank_wolfe_gap"],
    "time": data_short_step["timing"],
}
line_search_results = {
    "name": r"Line search",
    "f_value": data_line_seach["function_eval"],
    "primal_gap": [y - fValOpt for y in data_line_seach["function_eval"]],
    "dual_gap": data_line_seach["frank_wolfe_gap"],
    "time": data_line_seach["timing"],
}

away_results = {
    "name": r"Away",
    "f_value": data_away["function_eval"],
    "primal_gap": [y - fValOpt for y in data_away["function_eval"]],
    "dual_gap": data_away["frank_wolfe_gap"],
    "time": data_away["timing"],
}

away_results_ls = {
    "name": r"Away (line search)",
    "f_value": data_away_ls["function_eval"],
    "primal_gap": [y - fValOpt for y in data_away_ls["function_eval"]],
    "dual_gap": data_away_ls["frank_wolfe_gap"],
    "time": data_away_ls["timing"],
}

CGS_data = {
    "name": "CGS",
    "primal_gap": [y - fValOpt for y in CGS["function_eval"]],
    "frank_wolfe_gap": CGS["frank_wolfe_gap"],
    "FOO_calls": CGS["FOO_calls"],
    "LMO_calls": CGS["LMO_calls"],
    "timing": CGS["timing"],
}

experiment_details = {
    "dimension": dimension,
    "numSteps": numSteps,
    "Mu": Mu,
    "L": L,
    "feasible_region": "L1_ball",
}
results = {
    "details": experiment_details,
    "standard": standard_results,
    "short_step": short_step_results,
    "line_search": line_search_results,
    "CGS": CGS_data,
    "away": away_results,
    "away_ls": away_results_ls,
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
filepath = os.path.join(os.getcwd(), "output_data", "FW_CGS_comparison_str_cvx.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

output_filepath = os.path.join(
    output_directory_images,
    "FW_CGS_primal_gap_LMO_str_cvx.pdf",
)

list_x_label = [
    np.arange(len(results["standard"]["primal_gap"])) + 1,
    np.arange(len(results["short_step"]["primal_gap"])) + 1,
    np.arange(len(results["line_search"]["primal_gap"])) + 1,
    np.asarray(results["CGS"]["LMO_calls"]) + 1,
]
list_data = [
    results["standard"]["primal_gap"],
    results["short_step"]["primal_gap"],
    results["line_search"]["primal_gap"],
    results["CGS"]["primal_gap"],
]

colors = ["k", "b", "r", "c"]
markers = ["o", "s", "^", "P"]
colors_fill = ["None", "None", "None", "None"]

from frankwolfe.plotting_function import plot_results

figure_size = 7.3

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
            len(results["standard"]["primal_gap"]),
            len(results["short_step"]["primal_gap"]),
            len(results["line_search"]["primal_gap"]),
            results["CGS"]["LMO_calls"][-1],
        ),
    ],
    save_figure=output_filepath,
    # save_figure=None,
    figure_width=figure_size,
)

output_filepath = os.path.join(
    output_directory_images,
    "FW_CGS_primal_gap_FOO_str_cvx.pdf",
)

# Bound for GSC
num_points_bound = 58
values = np.zeros(num_points_bound)
values[0] = results["standard"]["dual_gap"][0]
values_x = np.zeros(num_points_bound)
values_x[0] = 0
for i in range(1, len(values)):
    values[i] = values[i - 1] / 2
    values_x[i] = values_x[i - 1] + int(
        np.ceil(2 * np.sqrt(6.0 * results["details"]["L"] / results["details"]["Mu"]))
    )


list_x_label = [
    np.arange(len(results["standard"]["primal_gap"])) + 1,
    np.arange(len(results["short_step"]["primal_gap"])) + 1,
    np.arange(len(results["line_search"]["primal_gap"])) + 1,
    np.asarray(results["CGS"]["FOO_calls"]) + 1,
    values_x,
]
list_data = [
    results["standard"]["primal_gap"],
    results["short_step"]["primal_gap"],
    results["line_search"]["primal_gap"],
    results["CGS"]["primal_gap"],
    values,
]
list_legend = [
    r"$\mathrm{FW \ (line \ search)}$",
    r"$\mathrm{FW \ (short \ step)}$",
    r"$\mathrm{FW \ } (2/(t + 2)$)",
    r"$\mathrm{CGS}$",
    r"$\mathrm{CGS \ FOO \ bound}$",
]


colors = ["k", "b", "r", "c", "--m"]
markers = ["o", "s", "^", "P", None]
colors_fill = ["None", "None", "None", "None", "None"]

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
            len(results["standard"]["primal_gap"]),
            len(results["short_step"]["primal_gap"]),
            len(results["line_search"]["primal_gap"]),
            results["CGS"]["FOO_calls"][-1],
        ),
    ],
    save_figure=output_filepath,
    # save_figure=None,
    figure_width=figure_size,
)
