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
    decomposition_invariant_pairwise_frank_wolfe,
    locally_accelerated_conditional_gradient_simplex,
)
from frankwolfe.feasible_regions import probability_simplex
from frankwolfe.objective_functions import quadratic_sparse_signal_recovery_probability_simplex

from frankwolfe.auxiliary_functions import (
    step_size_class,
    stopping_criterion,
)

# -----------------------L1 Ball example.------------------------------

numSteps = 100000

n = 500
m = 200
sigma = 0.01
non_zero_elements = 0.25
maximum_time = 300

x_true = np.zeros(n)
positions = np.random.choice(np.arange(n), int(non_zero_elements * n), replace=False)
x_true[positions] = np.random.rand(int(non_zero_elements * n)) - 0.5
A = np.random.rand(m, n)
y = A.dot(x_true) + np.random.normal(scale=sigma, size=m)


# function = quadratic_sparse_signal_recovery(A,y, alpha = 0.1)
# LPOracle = L1_ball(n, alpha = 0.95*np.sum(np.abs(x_true)))


function = quadratic_sparse_signal_recovery_probability_simplex(A, y, alpha=1.0)
# LPOracle = probability_simplex(int(2 * n), alpha=1.0 * np.sum(np.abs(x_true)))
LPOracle = probability_simplex(int(2 * n), alpha=30)

initial_point = LPOracle.initial_point()


ref_PFW_data = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "pairwise",
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap":1.0e-10}),
)
f_val_opt = ref_PFW_data["function_eval"][-1]

data_LaCG = locally_accelerated_conditional_gradient_simplex(
    function,
    LPOracle,
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": numSteps}),
)

output_BCG_data = blended_conditional_gradients(
    function,
    LPOracle,
    algorithm_parameters={
        "maintain_active_set": False,
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap":1.0e-8}),
)

output_PFW_data = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "pairwise",
        "maintain_active_set": False,
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap":1.0e-8}),
)

output_DICG_data = decomposition_invariant_pairwise_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "standard",
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap":1.0e-8}),
)

output_DICG_boosted_data = decomposition_invariant_pairwise_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "boosted",
        "boosting_delta": 1e-1,
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap":1.0e-8}),
)

output_AFW_data = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "standard",
        "maintain_active_set": False,
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap":1.0e-8}),
)

output_FW_data = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "standard",
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap":1.0e-8}),
)

output_FW_boosted_data = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "boosted",
        "boosting_delta": 1e-1,
    },
    x0 = initial_point,
    step_size = step_size_class("line_search"),
    stopping_criteria=stopping_criterion({"iterations": numSteps, "frank_wolfe_gap":1.0e-8}),
)


# Put the results in a picle object and then output
LaCG = {
    "name": "LaCG",
    "primal_value": data_LaCG["function_eval"],
    "primal_gap": [val - f_val_opt for val in data_LaCG["function_eval"]],
    "dual_gap": data_LaCG["frank_wolfe_gap"],
    "time": data_LaCG["timing"],
}

data_FW = {
    "name": "vanilla_FW",
    "primal_value": output_FW_data["function_eval"],
    "primal_gap": [val - f_val_opt for val in output_FW_data["function_eval"]],
    "dual_gap": output_FW_data["frank_wolfe_gap"],
    "time": output_FW_data["timing"],
}
data_AFW = {
    "name": "AFW",
    "primal_value": output_AFW_data["function_eval"],
    "primal_gap": [val - f_val_opt for val in output_AFW_data["function_eval"]],
    "dual_gap": output_AFW_data["frank_wolfe_gap"],
    "time": output_AFW_data["timing"],
}
data_PFW = {
    "name": "PFW",
    "primal_value": output_PFW_data["function_eval"],
    "primal_gap": [val - f_val_opt for val in output_PFW_data["function_eval"]],
    "dual_gap": output_PFW_data["frank_wolfe_gap"],
    "time": output_PFW_data["timing"],
}
data_boosted_FW = {
    "name": "boosted_FW",
    "primal_value": output_FW_boosted_data["function_eval"],
    "primal_gap": [val - f_val_opt for val in output_FW_boosted_data["function_eval"]],
    "LP_oracle_calls": output_FW_boosted_data["LP_oracle_calls"],
    "dual_gap": output_FW_boosted_data["frank_wolfe_gap"],
    "time": output_FW_boosted_data["timing"],
}
data_boosted_DICG = {
    "name": "Boosted_DICG",
    "primal_value": output_DICG_boosted_data["function_eval"],
    "primal_gap": [
        val - f_val_opt for val in output_DICG_boosted_data["function_eval"]
    ],
    "LP_oracle_calls": output_DICG_boosted_data["LP_oracle_calls"],
    "dual_gap": output_DICG_boosted_data["frank_wolfe_gap"],
    "time": output_DICG_boosted_data["timing"],
}
data_BCG = {
    "name": "BCG",
    "primal_value": output_BCG_data["function_eval"],
    "primal_gap": [val - f_val_opt for val in output_BCG_data["function_eval"]],
    "dual_gap": output_BCG_data["frank_wolfe_gap"],
    "time": output_BCG_data["timing"],
}
data_DICG = {
    "name": "DICG",
    "primal_value": output_DICG_data["function_eval"],
    "primal_gap": [val - f_val_opt for val in output_DICG_data["function_eval"]],
    "dual_gap": output_DICG_data["frank_wolfe_gap"],
    "time": output_DICG_data["timing"],
}

results = {
    "LaCG": LaCG,
    "vanilla_FW": data_FW,
    "AFW": data_AFW,
    "PFW": data_PFW,
    "boosted_FW": data_boosted_FW,
    "data_BCG": data_BCG,
    "DICG": data_DICG,
    "Boosted_DICG": data_boosted_DICG,
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
    os.getcwd(), "output_data", "boosted_comparison_version_sparsity25_v2.pickle"
)
with open(filepath, "wb") as f:
    pickle.dump(results, f)


figure_size = 7.3

# max_BCG_iterations = 4500
list_x_label = [
    np.arange(len(results["vanilla_FW"]["primal_gap"])) + 1,
    np.arange(
        len(
            np.asarray(results["AFW"]["primal_gap"])
        )
    )
    + 1,
    np.arange(
        len(
            np.asarray(results["PFW"]["primal_gap"])
        )
    )
    + 1,
    np.arange(
        len(
            np.asarray(results["data_BCG"]["primal_gap"])
        )
    )
    + 1,
    # np.arange(len(data["DICG"]["primal_gap"])) + 1,
    np.arange(
        len(
            np.asarray(results["LaCG"]["primal_gap"])
        )
    )
    + 1,
    # np.asarray(data["boosted_FW"]["LP_oracle_calls"]) + 1,
    # np.asarray(data["Boosted_DICG"]["LP_oracle_calls"]) + 1,
]

list_data = [
    results["vanilla_FW"]["primal_gap"],
    np.asarray(results["AFW"]["primal_gap"]),
    np.asarray(results["PFW"]["primal_gap"]),
    np.asarray(results["data_BCG"]["primal_gap"]),
    # data["DICG"]["primal_gap"],
    np.asarray(results["LaCG"]["primal_gap"]),
    # data["boosted_FW"]["primal_gap"],
    # data["Boosted_DICG"]["primal_gap"],
]

list_legend = [
    r"$\mathrm{FW}$",
    r"$\mathrm{AFW}$",
    r"$\mathrm{PFW}$",
    r"$\mathrm{BCG}$",
    # "DI-PFW",
    r"$\mathrm{LaCG}$",
    # "BoostedFW",
    # "BoostedDI-PFW"
]

from frankwolfe.plotting_function import plot_results

colors = [
    "k",
    "b",
    "y",
    "c",
    # "g",
    "tab:gray",
    # "r",
    # "m",
]
markers = [
    "o",
    "s",
    "D",
    "P",
    # "H",
    "*",
    # "^",
    # "X",
]
colors_fill = ["None", "None", "None", "None", "None", "None", "None", "None"]

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Number \ of \ LMO \ calls}$",
    r"$f(x_t) - f(x^*)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    x_limits=[1, None],
    y_limits=[1.0e-11, None],
    number_starting_markers_skip=7,
    # label_font_size=14.5,
    # number_columns_legend=2,
    save_figure=os.path.join(output_directory_images, "Boosted_BCG_FOO_25_v2.pdf"),
    figure_width=figure_size,
)

list_x_label = [
    np.arange(len(results["vanilla_FW"]["primal_gap"])) + 1,
    np.arange(
        len(
            np.asarray(results["AFW"]["primal_gap"])
        )
    )
    + 1,
    np.arange(
        len(
            np.asarray(results["PFW"]["primal_gap"])
        )
    )
    + 1,
    np.arange(
        len(
            np.asarray(results["data_BCG"]["primal_gap"])
        )
    )
    + 1,
    # np.arange(len(data["DICG"]["primal_gap"])) + 1,
    np.arange(
        len(
            np.asarray(results["LaCG"]["primal_gap"])
        )
    )
    + 1,
    # np.asarray(data["boosted_FW"]["LP_oracle_calls"]) + 1,
    # np.asarray(data["Boosted_DICG"]["LP_oracle_calls"]) + 1,
]


list_x_label = [
    results["vanilla_FW"]["time"],
    np.asarray(results["AFW"]["time"]),
    np.asarray(results["PFW"]["time"]),
    np.asarray(results["data_BCG"]["time"]),
    # data["DICG"]["time"],
    np.asarray(results["LaCG"]["time"]),
    # data["boosted_FW"]["time"],
    # data["Boosted_DICG"]["time"],
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
    y_limits=[1.0e-11, None],
    colorfills=colors_fill,
    log_x=False,
    log_y=True,
    number_starting_markers_skip=4,
    save_figure=os.path.join(output_directory_images, "Boosted_BCG_time_25_v2.pdf"),
    figure_width=figure_size,
)

