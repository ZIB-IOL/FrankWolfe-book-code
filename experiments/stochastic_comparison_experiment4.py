# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:24:18 2020

@author: pccom
"""
import numpy as np
import sys, os, time, datetime
import pickle

from frankwolfe.objective_functions import quadratic_stochastic
from frankwolfe.feasible_regions import box_constraints
from frankwolfe.algorithms import frank_wolfe
from frankwolfe.stochastic_algorithms import (
    stochastic_frank_wolfe,
)
from frankwolfe.auxiliary_functions import (
    stopping_criterion,
    step_size_class,
    stopping_criterion,
)

dimension = 100
radius = 5
lower_bound = -radius * np.ones(dimension)
upper_bound = radius * np.ones(dimension)
feasible_region = box_constraints(lower_bound, upper_bound)

xOpt = 20.0 * (np.random.rand(dimension) - 1.0)
sigma = 10
maximum_time = 7200
maximum_iterations = int(1.0e11)
minimum_eigenvalue_hessian = 0.0
maximum_eigenvalue_hessian = 100.0

function = quadratic_stochastic(
    dimension,
    xOpt,
    Mu=minimum_eigenvalue_hessian,
    L=maximum_eigenvalue_hessian,
    sigma=sigma,
)
G = np.sqrt(dimension) * sigma * (radius + 1)

initial_point = feasible_region.initial_point()
active_set = [initial_point]
barycentric_coordinates = [1.0]

if np.max(np.abs(xOpt)) <= radius:
    f_val_opt = function.f(xOpt)
else:
    results_FW = frank_wolfe(
        function,
        feasible_region,
        x0 = initial_point,
        step_size =step_size_class("line_search"),
        stopping_criteria=stopping_criterion({"iterations": maximum_iterations, "timing": maximum_time, "frank_wolfe_gap":1.0e-8}),
    )
    f_val_opt = results_FW["function_eval"][-1]

stopping_criteria = stopping_criterion({"timing": maximum_time,"iterations": maximum_iterations, "frank_wolfe_gap":1.0e-4})

results_stochastic = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria=stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "standard",
        "sigma": G,
    },
)

results_SCGS = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria=stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "conditional_gradient_sliding",
        "sigma": G,
    },
)

results_stochastic_SPIDER = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria=stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "variance_reduced_SPIDER",
        "sigma": G,
    },
)

results_stochastic_one_sample = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria = stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "one_sample",
        "frequency_output": 1000,
        "min_initial_output": 1000,
    },
)

results_stochastic_momentum = stochastic_frank_wolfe(
    function,
    feasible_region,
    x0 = initial_point.copy(),
    stopping_criteria = stopping_criteria,
    algorithm_parameters={
        "algorithm_type": "momentum",
        "frequency_output": 1000,
        "min_initial_output": 1000,
    },
)


experiment_details = {
    "dimension": dimension,
    "Mu": minimum_eigenvalue_hessian,
    "L": maximum_eigenvalue_hessian,
    "feasible_region": "box constraints",
    "radius": radius,
    "sigma": sigma,
    "maximum_iterations": maximum_iterations,
    "maximum_time": maximum_time,
    "unconstrained_opt": xOpt,
}

# Put the results in a picle object and then output
SFW_data = {
    "name": r"SFW",
    "stochastic_oracle_calls": results_stochastic["SFOO_calls"],
    "f_value": results_stochastic["function_eval"],
    "primal_gap": [y - f_val_opt for y in results_stochastic["function_eval"]],
    "dual_gap": results_stochastic["frank_wolfe_gap"],
    "time": results_stochastic["timing"],
}
MSFW_data = {
    "name": r"Momentum SFW",
    "stochastic_oracle_calls": results_stochastic_momentum["SFOO_calls"],
    "LMO_oracle_calls": results_stochastic_momentum["LMO_calls"],
    "f_value": results_stochastic_momentum["function_eval"],
    "primal_gap": [y - f_val_opt for y in results_stochastic_momentum["function_eval"]],
    "dual_gap": results_stochastic_momentum["frank_wolfe_gap"],
    "time": results_stochastic_momentum["timing"],
}
SPIDERFW = {
    "name": r"SPIDER FW",
    "stochastic_oracle_calls": results_stochastic_SPIDER["SFOO_calls"],
    "f_value": results_stochastic_SPIDER["function_eval"],
    "primal_gap": [y - f_val_opt for y in results_stochastic_SPIDER["function_eval"]],
    "dual_gap": results_stochastic_SPIDER["frank_wolfe_gap"],
    "time": results_stochastic_SPIDER["timing"],
}
SFW1_data = {
    "name": r"1-SFW",
    "stochastic_oracle_calls": results_stochastic_one_sample["SFOO_calls"],
    "LMO_oracle_calls": results_stochastic_one_sample["LMO_calls"],
    "f_value": results_stochastic_one_sample["function_eval"],
    "primal_gap": [
        y - f_val_opt for y in results_stochastic_one_sample["function_eval"]
    ],
    "dual_gap": results_stochastic_one_sample["frank_wolfe_gap"],
    "time": results_stochastic_one_sample["timing"],
}
SCGS = {
    "name": r"SCGS",
    "stochastic_oracle_calls": results_SCGS["SFOO_calls"],
    "f_value": results_SCGS["function_eval"],
    "primal_gap": [y - f_val_opt for y in results_SCGS["function_eval"]],
    "dual_gap": results_SCGS["frank_wolfe_gap"],
    "time": results_SCGS["timing"],
}

results = {
    "details": experiment_details,
    "SFW": SFW_data,
    "SFW (momentum)": MSFW_data,
    "SPIDER": SPIDERFW,
    "1-SFW": SFW1_data,
    "SCGS": SCGS,
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
    os.getcwd(),
    "output_data",
    "stoch_"
    + str(int(sigma))
    + "sig_"
    + str(dimension)
    + "dim_"
    + str(int(radius))
    + "rad_"
    + str(int(minimum_eigenvalue_hessian))
    + "mu_"
    + str(int(maximum_eigenvalue_hessian))
    + "L_"
    + str(timestamp)
    + ".pickle",
)
with open(filepath, "wb") as f:
    pickle.dump(results, f)

figure_size = 7.3
axis_font_size = 8
label_font_size = 8

list_x = [
    np.asarray(results["SFW"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["SPIDER"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["SFW (momentum)"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["1-SFW"]["stochastic_oracle_calls"]) + 1,
    np.asarray(results["SCGS"]["stochastic_oracle_calls"]) + 1,
]
list_data = [
    results["SFW"]["primal_gap"],
    results["SPIDER"]["primal_gap"],
    results["SFW (momentum)"]["primal_gap"],
    results["1-SFW"]["primal_gap"],
    results["SCGS"]["primal_gap"],
]
list_legend = [
    r"$\mathrm{SFW}$",
    r"$\mathrm{SPIDER}$",
    r"$\mathrm{M-SFW}$",
    r"$\mathrm{1-SFW}$",
    r"$\mathrm{SCGS}$",
]

x_limits = [1, None]

from frankwolfe.plotting_function import plot_results

colors = ["k", "b", "m", "c", "r", "g"]
markers = ["o", "s", "^", "P", "X", "H"]
colors_fill = ["None", "None", "None", "None", "None", "None", "None"]

output_filepath = os.path.join(
    output_directory_images,
    "new_SFOO_stoch_comparison_dim"
    + str(dimension)
    + "_sigma"
    + str(sigma)
    + "_radius"
    + str(radius)
    + "_mu"
    + str(minimum_eigenvalue_hessian)
    + "_L"
    + str(maximum_eigenvalue_hessian)
    + ".pdf",
)


plot_results(
    list_x,
    list_data,
    [],
    "",
    r"$\mathrm{Number \ of \ SFOO \ calls}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=x_limits,
    save_figure=output_filepath,
    figure_width=figure_size,
    axis_font_size=axis_font_size,
)

output_filepath = os.path.join(
    output_directory_images,
    "new_LMO_stoch_comparison_dim"
    + str(dimension)
    + "_sigma"
    + str(sigma)
    + "_radius"
    + str(radius)
    + "_mu"
    + str(minimum_eigenvalue_hessian)
    + "_L"
    + str(maximum_eigenvalue_hessian)
    + ".pdf",
)

list_x = [
    np.arange(len(results["SFW"]["primal_gap"])) + 1,
    np.arange(len(results["SPIDER"]["primal_gap"])) + 1,
    results["SFW (momentum)"]["LMO_oracle_calls"],
    results["1-SFW"]["LMO_oracle_calls"],
    np.arange(len(results["SCGS"]["primal_gap"])) + 1,
]

plot_results(
    list_x,
    list_data,
    # [],
    list_legend,
    "",
    r"$\mathrm{Number \ of \ LMO \ calls}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=x_limits,
    save_figure=output_filepath,
    figure_width=figure_size,
    axis_font_size=axis_font_size,
    label_font_size=label_font_size,
)

output_filepath = os.path.join(
    output_directory_images,
    "new_time_stoch_comparison_dim"
    + str(dimension)
    + "_sigma"
    + str(sigma)
    + "_radius"
    + str(radius)
    + "_mu"
    + str(minimum_eigenvalue_hessian)
    + "_L"
    + str(maximum_eigenvalue_hessian)
    + ".pdf",
)

list_x = [
    results["SFW"]["time"],
    results["SPIDER"]["time"],
    results["SFW (momentum)"]["time"],
    results["1-SFW"]["time"],
    results["SCGS"]["time"],
]

plot_results(
    list_x,
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
    save_figure=output_filepath,
    figure_width=figure_size,
    axis_font_size=axis_font_size,
)

