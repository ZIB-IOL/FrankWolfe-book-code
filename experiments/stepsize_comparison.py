# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:37:39 2020

@author: pccom
"""
import sys, os
import numpy as np
import pickle

from frankwolfe.algorithms import frank_wolfe, projected_gradient_descent
from frankwolfe.feasible_regions import L1_ball
from frankwolfe.objective_functions import quadratic_type_3
from frankwolfe.feasible_regions import L2_ball
from frankwolfe.auxiliary_functions import step_size_class, stopping_criterion

# -----------------------L1 Ball example.------------------------------

dimension = 500
tol = 1.0e-9
numSteps = 1000
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

# Run the standard stepsize.
data_standard_step_size = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("constant_step"),
)

# Run the short-step rule stepsize.
data_short_step = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("short_step"),
)

# Run the line search stepsize.
data_line_seach = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("line_search"),
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
}


output_directory_images = os.path.join(os.getcwd(), "output_images")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory_images):
    os.makedirs(output_directory_images)


output_directory = os.path.join(os.getcwd(), "output_data")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

# Output the results as a pickled object for later use.
filepath = os.path.join(os.getcwd(), "output_data", "step_size_results_L1_ball.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

from frankwolfe.plotting_function import plot_results

output_filepath = os.path.join(
    output_directory_images,
    "StepSize_Comparison_L1_"
    + str(dimension)
    + "_L"
    + str(int(L))
    + "_Mu"
    + str(int(Mu))
    + "_It"
    + str(numSteps)
    + ".pdf",
)

list_x_label = [
    np.arange(len(results["line_search"]["primal_gap"])) + 1,
    np.arange(len(results["short_step"]["primal_gap"])) + 1,
    np.arange(len(results["standard"]["primal_gap"])) + 1,
]
list_data = [
    results["line_search"]["primal_gap"],
    results["short_step"]["primal_gap"],
    results["standard"]["primal_gap"],
]
list_legend = [r"$\mathrm{Line \ search}$", r"$\mathrm{Short \ step}$", r"$2/(t + 2)$"]

figure_size = 7.3


colors = ["k", "b", "r"]
markers = ["o", "s", "^"]
colors_fill = ["None", "None", "None"]


plot_results(
    list_x_label,
    list_data,
    [],
    "",
    r"$\mathrm{Iteration \ (t)}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=[
        1,
        1
        + max(
            len(results["line_search"]["primal_gap"]),
            len(results["short_step"]["primal_gap"]),
            len(results["standard"]["primal_gap"]),
        ),
    ],
    save_figure=output_filepath,
    figure_width=figure_size,
)

# -----------------------L2 Ball example.------------------------------

dimension = 500
tol = 1.0e-9
numSteps = 1000
Mu = 1.0
L = 10000.0

LPOracle = L2_ball(dimension)
xOptGlobal = np.random.rand(dimension)
xOptGlobal /= np.linalg.norm(xOptGlobal)

function = quadratic_type_3(dimension, xOptGlobal, Mu=Mu, L=L)

initial_point = LPOracle.initial_point()

# Compute optimal solution using NAGD
opt = projected_gradient_descent(initial_point, function, LPOracle, tol)

fValOpt = function.f(opt)

# Run the standard stepsize.
data_standard_step_size = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("constant_step"),
)

# Run the short-step rule stepsize.
data_short_step = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("short_step"),
)

# Run the line search stepsize.
data_line_seach = frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
    step_size =step_size_class("line_search"),
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

experiment_details = {
    "dimension": dimension,
    "numSteps": numSteps,
    "Mu": Mu,
    "L": L,
    "feasible_region": "L2_ball",
}
results = {
    "details": experiment_details,
    "standard": standard_results,
    "short_step": short_step_results,
    "line_search": line_search_results,
}


# Output the results as a pickled object for later use.
filepath = os.path.join(os.getcwd(), "output_data", "step_size_results_L2_ball.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)


output_filepath = os.path.join(
    output_directory_images,
    "StepSize_Comparison_L2_"
    + str(dimension)
    + "_L"
    + str(int(L))
    + "_Mu"
    + str(int(Mu))
    + "_It"
    + str(numSteps)
    + ".pdf",
)

list_x_label = [
    np.arange(len(results["line_search"]["primal_gap"])) + 1,
    np.arange(len(results["short_step"]["primal_gap"])) + 1,
    np.arange(len(results["standard"]["primal_gap"])) + 1,
]
list_data = [
    results["line_search"]["primal_gap"],
    results["short_step"]["primal_gap"],
    results["standard"]["primal_gap"],
]

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
        1,
        1
        + max(
            len(results["line_search"]["primal_gap"]),
            len(results["short_step"]["primal_gap"]),
            len(results["standard"]["primal_gap"]),
        ),
    ],
    save_figure=output_filepath,
    figure_width=figure_size,
)