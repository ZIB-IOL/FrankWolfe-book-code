# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 08:03:23 2020

@author: pccom
"""
import numpy as np
import time
import pickle
import os
from scipy.sparse import issparse

# Define the stopping criteria for the algorithm.
class stopping_criterion:
    def __init__(self, dict_of_criteria):
        if type(dict_of_criteria) != dict:
            raise ValueError("Argument to stopping criteria must be a dictionary")
        if len(dict_of_criteria) == 0:
            raise ValueError("Argument to stopping criteria is empty")
        for key, value in dict_of_criteria.items():
            if key not in [
                "frank_wolfe_gap",
                "timing",
                "iterations",
            ]:
                raise ValueError(
                    "The dictionary of stopping criteria contains an unknown argument."
                )
            if value <= 0:
                raise ValueError(
                    "Argument for the stopping criteria for "
                    + key
                    + " must be greater than zero."
                )
        self.dict_of_criteria = dict_of_criteria
        if "iterations" not in dict_of_criteria.keys():
            self.dict_of_criteria["iterations"] = np.inf
        if "frank_wolfe_gap" not in dict_of_criteria.keys():
            self.dict_of_criteria["frank_wolfe_gap"] = -1
        if "timing" not in dict_of_criteria.keys():
            self.dict_of_criteria["timing"] = np.inf
        return

    def evaluate_stopping_criteria(self, data):
        return (
            data["frank_wolfe_gap"][-1] <= self.dict_of_criteria["frank_wolfe_gap"]
            or data["timing"][-1] >= self.dict_of_criteria["timing"]
        )

# Pick a stepsize.
class step_size_class:
    def __init__(self, type_of_step, step_size_parameters=None):
        list_valid_steps = [
            "adaptive_short_step",
            "short_step",
            "constant_step",
            "line_search",
        ]
        if type_of_step not in list_valid_steps:
            raise ValueError(
                "Argument to stopping criteria must be one of the following: "
                + str(list_valid_steps)
            )
        self.type_of_step = type_of_step
        if step_size_parameters is None:
            self.step_size_parameters = {}
        else:
            self.step_size_parameters = step_size_parameters
        if type_of_step == "constant_step":
            # If the constant stepsize parameters have not been set
            # use the default values
            if step_size_parameters is None or "additive_constant" not in step_size_parameters:
                self.step_size_parameters["additive_constant"] = 2
            if step_size_parameters is None or "multiplicative_constant" not in step_size_parameters:  
                self.step_size_parameters["multiplicative_constant"] = 2
        if type_of_step == "adaptive_short_step":
            # If the adaptive_short_step stepsize parameters have not been set
            # use the default values
            if step_size_parameters is None or "eta" not in step_size_parameters:
                self.step_size_parameters["eta"] =  0.9
            if step_size_parameters is None or "tau" not in step_size_parameters:  
                self.step_size_parameters["tau"] = 2.0
            self.step_size_parameters["L_estimate"] = 1.0e-2
        return

    def compute_step_size(self, function, x, d, grad, it, maximum_stepsize = 1.0):
        if self.type_of_step == "short_step":
            alpha = -grad.dot(d) / (
                function.largest_eigenvalue_hessian() * d.dot(d)
            )
        if self.type_of_step == "constant_step":
            alpha = self.step_size_parameters["multiplicative_constant"] / (
                it + self.step_size_parameters["additive_constant"]
            )
        if self.type_of_step == "line_search":
            alpha = function.line_search(x, d)
        if self.type_of_step == "adaptive_short_step":
            alpha, L_estimate = self.backtracking_step_size(
                function,
                d,
                x,
                grad,
                self.step_size_parameters["L_estimate"],
                maximum_stepsize,
                tau=self.step_size_parameters["tau"],
                eta=self.step_size_parameters["eta"],
            )
            self.step_size_parameters["L_estimate"] = L_estimate
        return min(alpha, maximum_stepsize)

    def backtracking_step_size(self, function, d, x, grad, L, alpha_max, tau, eta):
        M = L * eta
        d_norm_squared = np.dot(d, d)
        g_t = np.dot(-grad, d)
        alpha = min(g_t / (M * d_norm_squared), alpha_max)
        while (
            function.f(x + alpha * d)
            > function.f(x) - alpha * g_t + 0.5 * M * d_norm_squared * alpha * alpha
        ):
            M *= tau
            alpha = min(g_t / (M * d_norm_squared), alpha_max)
        return alpha, M


def calculate_stepsize_SIDO(x, d):
    arr = np.divide(x, d)
    index = np.where(arr > 0, arr, np.inf).argmin()
    return arr[index], index


def output_online_learning_matrix(
    matrix, G_estimate, sampling_order, radius_norm_ball, rank, users, movies
):
    data = {
        "matrix": matrix,
        "G_estimate": G_estimate,
        "sampling_order": sampling_order,
        "radius_norm_ball": radius_norm_ball,
        "rank": rank,
    }
    # Output the results as a pickled object for later use.
    filepath = os.path.join(
        os.getcwd(),
        "Data",
        "online_learning_matrix_"
        + str(radius_norm_ball)
        + "_"
        + str(rank)
        + "_"
        + str(users)
        + "_"
        + str(movies)
        + ".pickle",
    )
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def input_online_learning_matrix(radius_norm_ball, rank, users, movies):
    filepath = os.path.join(
        os.getcwd(),
        "Data",
        "online_learning_matrix_"
        + str(radius_norm_ball)
        + "_"
        + str(rank)
        + "_"
        + str(users)
        + "_"
        + str(movies)
        + ".pickle",
    )
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


# Generates a set of n-orthonormal vectors.
def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum()
        mat = np.eye(dim)
        mat[n - 1 :, n - 1 :] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


# def is_vextex_present(vertex, active_set):
#     for i in range(len(active_set)):
#         if np.allclose(vertex, active_set[i]):
#             return True, i
#     return False, np.nan


def short_circuit_check(a, b, n):
    L = int(np.floor(len(a) / n))
    for i in range(n):
        j = i * L
        if not all(a[j : j + L] == b[j : j + L]):
            return False
    if len(a) % n != 0:
        # Check the last chunk.
        if not all(a[j + L :] == b[j + L :]):
            return False
    return True


def is_vextex_present(vertex, active_set):
    if issparse(active_set[0]) and issparse(vertex):
        for i in range(len(active_set)):
            if (vertex != active_set[i]).nnz == 0:
                return True, i
        return False, np.nan
    else:
        if issparse(active_set[0]):
            for i in range(len(active_set)):
                if short_circuit_check(vertex, active_set[i].todense(), 100):
                    return True, i
            return False, np.nan
        else:
            for i in range(len(active_set)):
                if short_circuit_check(vertex, active_set[i], 100):
                    return True, i
            return False, np.nan


def perform_update(
    x,
    v,
    grad,
    function,
    data,
    it,
    time_elapsed,
    return_points=True,
    monitor_frank_wolfe_gap=True,
):
    if monitor_frank_wolfe_gap:
        data["frank_wolfe_gap"].append(np.dot(grad, x - v))
    data["function_eval"].append(function.f(x))
    data["timing"].append(time_elapsed + data["timing"][-1])
    if return_points:
        data["x_val"].append(x.copy())
    return


# Calculates the maximum stepsize that we can take from 0 to 1.
# x + alpha*d
def calculate_stepsize_DIPCG(x, d):
    return np.min([-x[i] / d[i] if d[i] < 0 else np.inf for i in range(len(x))])

    # x = np.clip(x, 0, None)
    # index = np.where(x == 0)[0]
    # if np.any(d[index] < 0.0):
    #     return 0.0

    # index = np.where(x > 0)[0]
    # coeff = np.zeros(len(x))
    # for i in index:
    #     if d[i] < 0.0:
    #         coeff[i] = -x[i] / d[i]
    #     else:
    #         coeff[i] = np.inf
    # return min(coeff)

    # val = coeff[coeff > 0]
    # if len(val) == 0:
    #     print("No suitable direction")
    #     return 0.0
    # else:
    #     return min(val)
    # return min(val)


# # Pick a stepsize.
# def step_size(function, x, d, grad, it, step_size_param):
#     if step_size_param["type_step"] == "short_step":
#         alpha = -np.dot(grad, d) / (
#             function.largest_eigenvalue_hessian() * np.dot(d, d)
#         )
#     if step_size_param["type_step"] == "polyak_short_step":
#         alpha = (function.f(x) - step_size_param["f_optimum"]) / (
#             function.largest_eigenvalue_hessian() * np.dot(d, d)
#         )
#     if step_size_param["type_step"] == "constant_step":
#         if (
#             "additive_constant" in step_size_param
#             and "multiplicative_constant" in step_size_param
#         ):
#             alpha = step_size_param["multiplicative_constant"] / (
#                 it + step_size_param["additive_constant"]
#             )
#         else:
#             alpha = 2.0 / (it + 2.0)
#     if step_size_param["type_step"] == "line_search":
#         alpha = function.line_search(x, d)
#     if step_size_param["type_step"] == "adaptive_short_step":
#         alpha, L_estimate = backtracking_step_size(
#             function,
#             d,
#             x,
#             grad,
#             step_size_param["L_estimate"],
#             step_size_param["alpha_max"],
#             tau=step_size_param["tau"],
#             eta=step_size_param["eta"],
#         )
#         step_size_param["L_estimate"] = L_estimate
#     return min(alpha, step_size_param["alpha_max"])





# Provides an initial estimate for the smoothness parameter.
def smoothness_estimate(x0, function):
    L = 1.0e-3
    while function.f(x0 - function.grad(x0) / L) > function.f(x0):
        L *= 1.5
    return L


def max_min_vertex(grad, active_set):
    maxProd = np.dot(active_set[0], grad)
    minProd = np.dot(active_set[0], grad)
    maxInd = 0
    minInd = 0
    for i in range(len(active_set)):
        if np.dot(active_set[i], grad) > maxProd:
            maxProd = np.dot(active_set[i], grad)
            maxInd = i
        else:
            if np.dot(active_set[i], grad) < minProd:
                minProd = np.dot(active_set[i], grad)
                minInd = i
    return active_set[maxInd], maxInd, active_set[minInd], minInd


def max_min_vertex_quick_exit(feasible_region, grad, x, active_set, phi, K):
    for i in range(len(active_set)):
        if np.dot(grad, active_set[i] - x) >= phi / K:
            return active_set[i], i, None, None
        if np.dot(grad, x - active_set[i]) >= phi / K:
            return None, None, active_set[i], i
    v = feasible_region.linear_optimization_oracle(grad)
    return None, None, v, None


def brute_force_away_oracle(grad, active_set):
    max_product = np.dot(active_set[0], grad)
    max_point = active_set[0]
    index_max = 0
    for i in range(1, len(active_set)):
        if max_product < np.dot(active_set[i], grad):
            max_product = np.dot(active_set[i], grad)
            max_point = active_set[i]
            index_max = i
    return max_point, index_max
