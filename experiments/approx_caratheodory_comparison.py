# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:37:39 2020

@author: pccom
"""
import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from frankwolfe.algorithms import (
    fully_corrective_frank_wolfe,
    frank_wolfe,
    away_frank_wolfe,
    blended_conditional_gradients,
)
from frankwolfe.feasible_regions import birkhoff_polytope
from frankwolfe.objective_functions import quadratic
from frankwolfe.auxiliary_functions import step_size_class, stopping_criterion

# -----------------------L1 Ball example.------------------------------

dimension = 3600
mat_dimension = int(np.sqrt(dimension))
tol = 1.0e-9
numSteps = 10000000000

LPOracle = birkhoff_polytope(dimension)

num_matrices_used = 30
U = np.zeros(num_matrices_used + 1)
U[1:-1] = sorted(np.random.rand(num_matrices_used - 1))
U[-1] = 1
V = np.diff(U)
true_matrices = []

# Generate a vector as a convex combination of permutation matrices
vector = np.zeros((mat_dimension, mat_dimension))
for i in range(num_matrices_used):
    true_matrices.append(
        LPOracle.linear_optimization_oracle(np.random.rand(dimension) - 1).reshape(
            (mat_dimension, mat_dimension)
        )
    )
    vector += V[i] * true_matrices[-1]
vector = vector.flatten()
matrix = np.eye(dimension)
function = quadratic(matrix, -vector)

initial_point = LPOracle.initial_point()
fValOpt = -0.5 * np.linalg.norm(vector) ** 2

line_search = {"type_step": "line_search"}

maximum_time = 7200
# maximum_time = 300


def cardinality_log(point, convex_decomposition, objective_function, feasible_region):
    return len(convex_decomposition)


# Run the short-step rule stepsize.
FW_fully_corrective_data = fully_corrective_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-8, "timing": maximum_time,"iterations": numSteps}),
    logging_functions={"cardinality": cardinality_log},
)

# Run the short-step rule stepsize.
FW_away_lazy_data = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "lazy",
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-8, "timing": maximum_time,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
    logging_functions={"cardinality": cardinality_log},
)

FW_vanilla_lazy_data = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "lazy",
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-8, "timing": maximum_time,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
    logging_functions={"cardinality": cardinality_log},
)

FW_vanilla_data = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "standard",
        "maintain_active_set": True,
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-8, "timing": maximum_time,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
    logging_functions={"cardinality": cardinality_log},
)


FW_pairwise_data = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "pairwise",
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-8, "timing": maximum_time,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
    logging_functions={"cardinality": cardinality_log},
)

FW_away_data = away_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "standard",
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-8, "timing": maximum_time,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
    logging_functions={"cardinality": cardinality_log},
)


FW_NEP_simplex_data = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "nep",
        "maintain_active_set": True,
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-8, "timing": maximum_time,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
    logging_functions={"cardinality": cardinality_log},
)

output_BCG_data = blended_conditional_gradients(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-8, "timing": maximum_time,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
    logging_functions={"cardinality": cardinality_log},
)

# Put the results in a picle object and then output
pairwise_FW_pickle = {
    "name": "PFW",
    "primal_value": FW_pairwise_data["function_eval"],
    "primal_gap": [val - fValOpt for val in FW_pairwise_data["function_eval"]],
    "dual_gap": FW_pairwise_data["frank_wolfe_gap"],
    "cardinality": FW_pairwise_data["cardinality"],
    "time": FW_pairwise_data["timing"],
}
FW_pickle = {
    "name": "FW",
    "primal_value": FW_vanilla_data["function_eval"],
    "primal_gap": [val - fValOpt for val in FW_vanilla_data["function_eval"]],
    "dual_gap": FW_vanilla_data["frank_wolfe_gap"],
    "cardinality": FW_vanilla_data["cardinality"],
    "time": FW_vanilla_data["timing"],
}
away_FW_pickle = {
    "name": "AFW",
    "primal_value": FW_away_data["function_eval"],
    "primal_gap": [val - fValOpt for val in FW_away_data["function_eval"]],
    "dual_gap": FW_away_data["frank_wolfe_gap"],
    "cardinality": FW_away_data["cardinality"],
    "time": FW_away_data["timing"],
}
FCFW_pickle = {
    "name": "FCFW",
    "primal_value": FW_fully_corrective_data["function_eval"],
    "primal_gap": [val - fValOpt for val in FW_fully_corrective_data["function_eval"]],
    "dual_gap": FW_fully_corrective_data["frank_wolfe_gap"],
    "cardinality": FW_fully_corrective_data["cardinality"],
    "time": FW_fully_corrective_data["timing"],
}
away_lazy_FW_pickle = {
    "name": "LAFW",
    "primal_value": FW_away_lazy_data["function_eval"],
    "primal_gap": [val - fValOpt for val in FW_away_lazy_data["function_eval"]],
    "dual_gap": FW_away_lazy_data["frank_wolfe_gap"],
    "cardinality": FW_away_lazy_data["cardinality"],
    "time": FW_away_lazy_data["timing"],
}
FW_lazy_pickle = {
    "name": "LFW",
    "primal_value": FW_vanilla_lazy_data["function_eval"],
    "primal_gap": [val - fValOpt for val in FW_vanilla_lazy_data["function_eval"]],
    "dual_gap": FW_vanilla_lazy_data["frank_wolfe_gap"],
    "cardinality": FW_vanilla_lazy_data["cardinality"],
    "time": FW_vanilla_lazy_data["timing"],
}
NEP_FW_pickle = {
    "name": "NEPFW",
    "primal_value": FW_NEP_simplex_data["function_eval"],
    "primal_gap": [val - fValOpt for val in FW_NEP_simplex_data["function_eval"]],
    "dual_gap": FW_NEP_simplex_data["frank_wolfe_gap"],
    "cardinality": FW_NEP_simplex_data["cardinality"],
    "time": FW_NEP_simplex_data["timing"],
}
BCG_pickle = {
    "name": "BCG",
    "primal_value": output_BCG_data["function_eval"],
    "primal_gap": [val - fValOpt for val in output_BCG_data["function_eval"]],
    "dual_gap": output_BCG_data["frank_wolfe_gap"],
    "cardinality": output_BCG_data["cardinality"],
    "time": output_BCG_data["timing"],
}

results = {
    "dimension": dimension,
    "sparsity": num_matrices_used,
    "PFW": pairwise_FW_pickle,
    "FW": FW_pickle,
    "AFW": away_FW_pickle,
    "FCFW": FCFW_pickle,
    "LAFW": away_lazy_FW_pickle,
    "LFW": FW_lazy_pickle,
    "NEPFW": NEP_FW_pickle,
    "BCG": BCG_pickle,
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
filepath = os.path.join(os.getcwd(), "output_data", "approximate_caratheodory.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

list_legend = [
    r"$\mathrm{LCG}$",
    r"$\mathrm{Lazy \ AFW}$",
    r"$\mathrm{FCFW}$",
    r"$\mathrm{NEP-FW}$",
    r"$\mathrm{BCG}$",
    r"$\mathrm{PFW}$",
    r"$\mathrm{FW}$",
    r"$\mathrm{AFW}$",
]

from frankwolfe.plotting_function import plot_results

colors = ["k", "b", "y", "m", "c", "r", "g", "tab:gray"]
markers = ["o", "s", "D", "^", "P", "X", "H", "*"]
colors_fill = ["None", "None", "None", "None", "None", "None", "None", "None"]

figure_size = 7.3

list_data = [
    results["LFW"]["primal_gap"],
    results["LAFW"]["primal_gap"],
    results["FCFW"]["primal_gap"],
    results["NEPFW"]["primal_gap"],
    results["BCG"]["primal_gap"],
    # data["PFW"]["primal_gap"],
    # data["FW"]["primal_gap"],
    # data["AFW"]["primal_gap"],
]

list_x_label = [
    results["LFW"]["time"],
    results["LAFW"]["time"],
    results["FCFW"]["time"],
    results["NEPFW"]["time"],
    results["BCG"]["time"],
    # data["PFW"]["time"],
    # data["FW"]["time"],
    # data["AFW"]["time"],
]

plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Time \ (s)}$",
    r"$\mathrm{Distance \ to \ } u$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    # label_font_size=15,
    number_starting_markers_skip=7,
    # y_limits=[1.0e-8, None],
    # outside_legend = True,
    save_figure=os.path.join(
        output_directory_images, "caratheodory_primal_gap_time_v2.pdf"
    ),
    figure_width=figure_size,
)

list_x_label = [
    results["LFW"]["primal_gap"],
    results["LAFW"]["primal_gap"],
    results["FCFW"]["primal_gap"],
    results["NEPFW"]["primal_gap"],
    results["BCG"]["primal_gap"],
    # data["PFW"]["primal_gap"],
    # data["FW"]["primal_gap"],
    # data["AFW"]["primal_gap"],
]

list_data = [
    results["LFW"]["cardinality"],
    results["LAFW"]["cardinality"],
    results["FCFW"]["cardinality"],
    results["NEPFW"]["cardinality"],
    results["BCG"]["cardinality"],
    # data["PFW"]["cardinality"],
    # data["FW"]["cardinality"],
    # data["AFW"]["cardinality"],
]

plot_results(
    list_data,
    list_x_label,
    [],
    "",
    r"$\mathrm{Sparsity}$",
    r"$\mathrm{Distance \ to \ } u$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    # label_font_size=15,
    number_starting_markers_skip=7,
    # outside_legend = True,
    save_figure=os.path.join(
        output_directory_images, "caratheodory_cardinality_primal_gap_v2.pdf"
    ),
    figure_width=figure_size,
)

list_x_label = [
    results["LFW"]["time"],
    results["LAFW"]["time"],
    results["FCFW"]["time"],
    results["NEPFW"]["time"],
    results["BCG"]["time"],
    # data["PFW"]["primal_gap"],
    # data["FW"]["primal_gap"],
    # data["AFW"]["primal_gap"],
]

list_data = [
    results["LFW"]["cardinality"],
    results["LAFW"]["cardinality"],
    results["FCFW"]["cardinality"],
    results["NEPFW"]["cardinality"],
    results["BCG"]["cardinality"],
    # data["PFW"]["cardinality"],
    # data["FW"]["cardinality"],
    # data["AFW"]["cardinality"],
]

plot_results(
    list_x_label,
    list_data,
    [],
    "",
    r"$\mathrm{Time \ (s)}$",
    r"$\mathrm{Sparsity}$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    # label_font_size=15,
    number_starting_markers_skip=4,
    # outside_legend = True,
    save_figure=os.path.join(
        output_directory_images, "caratheodory_cardinality_time_v2.pdf"
    ),
    figure_width=figure_size,
)

