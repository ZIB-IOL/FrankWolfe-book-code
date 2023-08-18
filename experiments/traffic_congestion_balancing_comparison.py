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
    blended_conditional_gradients,
)
from frankwolfe.feasible_regions import gurobi_polytope
from frankwolfe.objective_functions import quadratic_diagonal

from frankwolfe.auxiliary_functions import (
    step_size_class,
    stopping_criterion,
)

numSteps = 50000

# Second feasible region.
file = "road_paths_01_DC_a.lp"
tolerance = 2e-1
Mu = 1
L = 1000

pathfile = os.path.join(os.getcwd(), "Data", "trafficnetwork", file)
LPOracle = gurobi_polytope(pathfile)

line_search = {"type_step": "line_search"}
initial_point = LPOracle.initial_point()
dimension = len(initial_point)


function = quadratic_diagonal(dimension, np.random.rand(dimension), Mu=Mu, L=L)

data_BCG_ref_point = blended_conditional_gradients(
    function,
    LPOracle,
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 72000.0, "iterations": int(10 * numSteps), "frank_wolfe_gap":1.0e-13}),
)

f_val_opt = function.f(data_BCG_ref_point["solution"])

data_BCG = blended_conditional_gradients(
    function,
    LPOracle,
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 36000.0, "iterations": numSteps, "frank_wolfe_gap":1.0e-13}),
)

AFW_data = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "type_step": "standard",
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 36000.0, "iterations": numSteps, "frank_wolfe_gap":1.0e-13}),
)

PFW_data = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "type_step": "pairwise",
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 36000.0, "iterations": numSteps, "frank_wolfe_gap":1.0e-13}),
)

FW_vanilla_data = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "type_step": "standard",
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 36000.0, "iterations": numSteps, "frank_wolfe_gap":1.0e-13}),
)

data_boosted_6 = frank_wolfe(
    function,
    LPOracle,
    numSteps,
    algorithm_parameters={
        "type_step": "boosted",
        "boosting_delta": 1e-6,
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 36000.0, "iterations": numSteps, "frank_wolfe_gap":1.0e-13}),
)

data_boosted_5 = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "type_step": "boosted",
        "boosting_delta": 1e-5,
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 36000.0, "iterations": numSteps, "frank_wolfe_gap":1.0e-13}),
)

data_boosted_4 = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "type_step": "boosted",
        "boosting_delta": 1e-4,
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 36000.0, "iterations": numSteps, "frank_wolfe_gap":1.0e-13}),
)

data_boosted_3 = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "type_step": "boosted",
        "boosting_delta": 1e-3,
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"timing": 36000.0, "iterations": numSteps, "frank_wolfe_gap":1.0e-13}),
)

# Put the results in a picle object and then output
vanilla_FW = {
    "name": "vanilla_FW",
    "primal_value": FW_vanilla_data["function_eval"],
    "primal_gap": [f - f_val_opt for f in FW_vanilla_data["function_eval"]],
    "dual_gap": FW_vanilla_data["frank_wolfe_gap"],
    "time": FW_vanilla_data["timing"],
}
data_AFW = {
    "name": "AFW",
    "primal_value": AFW_data["function_eval"],
    "primal_gap": [f - f_val_opt for f in AFW_data["function_eval"]],
    "dual_gap": AFW_data["frank_wolfe_gap"],
    "time": AFW_data["timing"],
}
data_PFW = {
    "name": "PFW",
    "primal_value": PFW_data["function_eval"],
    "primal_gap": [f - f_val_opt for f in PFW_data["function_eval"]],
    "dual_gap": PFW_data["frank_wolfe_gap"],
    "time": PFW_data["timing"],
}
boosted_FW_6 = {
    "name": "boosted_FW (delta 10e-6)",
    "primal_value": data_boosted_6["function_eval"],
    "primal_gap": [f - f_val_opt for f in data_boosted_6["function_eval"]],
    "LP_oracle_calls": data_boosted_6["LP_oracle_calls"],
    "dual_gap": data_boosted_6["frank_wolfe_gap"],
    "time": data_boosted_6["timing"],
}
boosted_FW_5 = {
    "name": "boosted_FW (delta 10e-5)",
    "primal_value": data_boosted_5["function_eval"],
    "primal_gap": [f - f_val_opt for f in data_boosted_5["function_eval"]],
    "LP_oracle_calls": data_boosted_5["LP_oracle_calls"],
    "dual_gap": data_boosted_5["frank_wolfe_gap"],
    "time": data_boosted_5["timing"],
}
boosted_FW_4 = {
    "name": "boosted_FW (delta 10e-4)",
    "primal_value": data_boosted_4["function_eval"],
    "primal_gap": [f - f_val_opt for f in data_boosted_4["function_eval"]],
    "LP_oracle_calls": data_boosted_4["LP_oracle_calls"],
    "dual_gap": data_boosted_4["frank_wolfe_gap"],
    "time": data_boosted_4["timing"],
}
boosted_FW_3 = {
    "name": "boosted_FW (delta 10e-3)",
    "primal_value": data_boosted_3["function_eval"],
    "primal_gap": [f - f_val_opt for f in data_boosted_3["function_eval"]],
    "LP_oracle_calls": data_boosted_3["LP_oracle_calls"],
    "dual_gap": data_boosted_3["frank_wolfe_gap"],
    "time": data_boosted_3["timing"],
}
BCG = {
    "name": "BCG",
    "primal_value": data_BCG["function_eval"],
    "primal_gap": [f - f_val_opt for f in data_BCG["function_eval"]],
    "dual_gap": data_BCG["frank_wolfe_gap"],
    "time": data_BCG["timing"],
}

results = {
    "vanilla_FW": vanilla_FW,
    "AFW": data_AFW,
    "PFW": data_PFW,
    "boosted_FW_6": boosted_FW_6,
    "boosted_FW_5": boosted_FW_5,
    "boosted_FW_4": boosted_FW_4,
    "boosted_FW_3": boosted_FW_3,
    "BCG": BCG,
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
filepath = os.path.join(
    os.getcwd(),
    "output_data",
    "traffic_congestion_balancing.pickle",
)
with open(filepath, "wb") as f:
    pickle.dump(results, f)


max_BCG_iterations = -1
list_x_label = [
    np.arange(len(results["vanilla_FW"]["primal_gap"])) + 1,
    np.arange(len(results["AFW"]["primal_gap"])) + 1,
    # np.arange(len(data["boosted_FW_6"]["primal_gap"])) + 1,
    np.arange(len(results["boosted_FW_5"]["primal_gap"])) + 1,
    # np.arange(len(data["boosted_FW_4"]["primal_gap"])) + 1,
    # np.arange(len(data["boosted_FW_3"]["primal_gap"])) + 1,
    (np.arange(len(results["BCG"]["primal_gap"])) + 1),
]
list_data = [
    results["vanilla_FW"]["primal_gap"],
    results["AFW"]["primal_gap"],
    # data["boosted_FW_6"]["primal_gap"],
    results["boosted_FW_5"]["primal_gap"],
    # data["boosted_FW_4"]["primal_gap"],
    # data["boosted_FW_3"]["primal_gap"],
    results["BCG"]["primal_gap"],
]
list_legend = [
    r"$\mathrm{FW}$",
    r"$\mathrm{AFW}$",
    # r"BoostFW $\left(\delta = 10^{-6}\right)$",
    # r"BoostFW $\left(\delta = 10^{-5}\right)$",
    r"$\mathrm{BoostFW}$",
    # r"BoostFW $\left(\delta = 10^{-4}\right)$",
    # r"BoostFW $\left(\delta = 10^{-3}\right)$",
    r"$\mathrm{BCG}$",
]

from frankwolfe.plotting_function import plot_results

colors = ["k", "b", "m", "c", "r", "g", "y"]
markers = ["o", "s", "^", "P", "X", "H", "D"]
colors_fill = ["None", "None", "None", "None", "None", "None", "None"]

figure_size = 7.3

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
    # number_columns_legend=2,
    x_limits=[1, None],
    # y_limits=[1.0e-8, None],
    save_figure=os.path.join(output_directory_images, "traffic_congestion_FOO.pdf"),
    # save_figure=None,
    figure_width=figure_size,
)

list_x_label = [
    np.arange(len(results["vanilla_FW"]["primal_gap"])) + 1,
    np.arange(len(results["AFW"]["primal_gap"])) + 1,
    # np.asarray(data["boosted_FW_6"]["LP_oracle_calls"]) + 1,
    np.asarray(results["boosted_FW_5"]["LP_oracle_calls"]) + 1,
    # np.asarray(data["boosted_FW_4"]["LP_oracle_calls"]) + 1,
    # np.asarray(data["boosted_FW_3"]["LP_oracle_calls"]) + 1,
    (np.arange(len(results["BCG"]["primal_gap"])) + 1),
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
    x_limits=[1, None],
    # y_limits=[1.0e-8, None],
    save_figure=os.path.join(output_directory_images, "traffic_congestion_LMO.pdf"),
    # save_figure=None,
    figure_width=figure_size,
)

list_x_label = [
    results["vanilla_FW"]["time"],
    results["AFW"]["time"],
    # data["boosted_FW_6"]["time"],
    results["boosted_FW_5"]["time"],
    # data["boosted_FW_4"]["time"],
    # data["boosted_FW_3"]["time"],
    (results["BCG"]["time"]),
]

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
    label_font_size=15,
    # y_limits=[1.0e-8, None],
    # outside_legend = True,
    save_figure=os.path.join(output_directory_images, "traffic_congestion_time.pdf"),
    # save_figure=None,
    figure_width=figure_size,
)
