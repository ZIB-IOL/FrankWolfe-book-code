# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:37:39 2020

@author: pccom
"""
import sys, os
import scipy.io
import pickle


from frankwolfe.algorithms import (
    frank_wolfe,
    away_frank_wolfe,
    blended_conditional_gradients,
    decomposition_invariant_pairwise_frank_wolfe,
)
from frankwolfe.feasible_regions import flow_polytope
from frankwolfe.objective_functions import quadratic
from frankwolfe.auxiliary_functions import step_size_class, stopping_criterion

# -----------------------L1 Ball example.------------------------------

data = scipy.io.loadmat(
    os.path.join(os.getcwd(), "Data", "videocolocalization", "aeroplane_data_small.mat")
)
A = data["A"]
b = data["b"]

numSteps = 500000
number_nodes_per_layer = 20
number_of_layers = 33

function = quadratic(A, b.flatten())
LPOracle = flow_polytope(number_nodes_per_layer, number_of_layers)

initial_point = LPOracle.initial_point()

data_DICG_ref_point = decomposition_invariant_pairwise_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-16, "timing": 600.0,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
)

f_val_opt = function.f(data_DICG_ref_point["solution"])

data_DICG = decomposition_invariant_pairwise_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-15, "timing": 600.0,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
)

data_DICG_Boosted = decomposition_invariant_pairwise_frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "boosted",
        "boosting_delta": 1e-15,
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-16, "timing": 600.0,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
)

data_BCG = blended_conditional_gradients(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-16, "timing": 600.0,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
)

AFW_data = away_frank_wolfe(
    function,
    LPOracle,
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-13, "timing": 600.0,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
)

FW_vanilla = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "standard",
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-13, "timing": 600.0,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
)

data_boosted = frank_wolfe(
    function,
    LPOracle,
    algorithm_parameters={
        "algorithm_type": "boosted",
        "boosting_delta": 1e-15,
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"frank_wolfe_gap": 1.0e-13, "timing": 600.0,"iterations": numSteps}),
    step_size =step_size_class("line_search"),
)


# Put the results in a picle object and then output
vanilla_FW = {
    "name": "vanilla_FW",
    "primal_value": FW_vanilla["function_eval"],
    "primal_gap": [f - f_val_opt for f in FW_vanilla["function_eval"]],
    "dual_gap": FW_vanilla["frank_wolfe_gap"],
    "time": FW_vanilla["timing"],
}
data_AFW = {
    "name": "AFW",
    "primal_value": AFW_data["function_eval"],
    "primal_gap": [f - f_val_opt for f in AFW_data["function_eval"]],
    "dual_gap": AFW_data["frank_wolfe_gap"],
    "time": AFW_data["timing"],
}
boosted_FW = {
    "name": "boosted_FW",
    "primal_value": data_boosted["function_eval"],
    "primal_gap": [f - f_val_opt for f in data_boosted["function_eval"]],
    "LP_oracle_calls": data_boosted["LP_oracle_calls"],
    "dual_gap": data_boosted["frank_wolfe_gap"],
    "time": data_boosted["timing"],
}
boosted_DICG = {
    "name": "Boosted_DICG",
    "primal_value": data_DICG_Boosted["function_eval"],
    "primal_gap": [f - f_val_opt for f in data_DICG_Boosted["function_eval"]],
    "LP_oracle_calls": data_DICG_Boosted["LP_oracle_calls"],
    "dual_gap": data_DICG_Boosted["frank_wolfe_gap"],
    "time": data_DICG_Boosted["timing"],
}
BCG = {
    "name": "BCG",
    "primal_value": data_BCG["function_eval"],
    "primal_gap": [f - f_val_opt for f in data_BCG["function_eval"]],
    "dual_gap": data_BCG["frank_wolfe_gap"],
    "time": data_BCG["timing"],
}
data_DICG = {
    "name": "SICG",
    "primal_value": data_DICG["function_eval"],
    "primal_gap": [f - f_val_opt for f in data_DICG["function_eval"]],
    "dual_gap": data_DICG["frank_wolfe_gap"],
    "time": data_DICG["timing"],
}


results = {
    "vanilla_FW": vanilla_FW,
    "AFW": data_AFW,
    "boosted_FW": boosted_FW,
    "BCG": BCG,
    "DICG": data_DICG,
    "Boosted_DICG": boosted_DICG,
}

# Output the results as a pickled object for later use.
filepath = os.path.join(
    os.getcwd(),
    "Data",
    "boosted_comparison_version_sparsity_videocolocalization.pickle",
)
with open(filepath, "wb") as f:
    pickle.dump(results, f)
