import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import math
from numba import jit
import itertools

from .auxiliary_functions import (
    calculate_stepsize_SIDO,
    perform_update,
    is_vextex_present,
    calculate_stepsize_DIPCG,
    max_min_vertex,
    max_min_vertex_quick_exit,
    stopping_criterion,
    step_size_class,
    # step_size,
)

from .objective_functions import (
    projection_problem_function,
    quadratic_function_over_simplex,
)
from .feasible_regions import probability_simplex, polytope_defined_by_vertices
from .objective_functions import quadratic_function_over_simplex

"""
----------------------------------------------------------------------
                                Algorithms 
----------------------------------------------------------------------
"""

def accelerated_projected_gradient_descent(
    x0,
    function,
    feasible_region,
    tolerance,
):
    """
    Run Nesterov's accelerated projected gradient descent.

    References
    ----------
    Nesterov, Y. (2018). Lectures on convex optimization (Vol. 137).
    Berlin, Germany: Springer. (Constant scheme II, Page 93)

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array.
    feasible_region : feasible region function.
        Returns projection oracle of a point x onto the feasible region,
        which are computed through the function feasible_region.project(x).
        Additionally, a LMO is used to compute the Frank-Wolfe gap (used as a
        stopping criterion) through the function
        feasible_region.linear_optimization_oracle(grad) function, which
        minimizes <x, grad> over the feasible region.
    tolerance : float
        Frank-Wolfe accuracy to which the solution is outputted.

    Returns
    -------
    x : numpy array
        Outputted solution with primal gap below the target tolerance
    """
    from collections import deque

    L = function.largest_eigenvalue_hessian()
    mu = function.smallest_eigenvalue_hessian()
    x = deque([x0], maxlen=2)
    y = x0
    q = mu / L
    alpha = deque([np.sqrt(q)], maxlen=2)
    grad = function.grad(x[-1])
    while (
        grad.dot(x[-1] - feasible_region.linear_optimization_oracle(grad)) > tolerance
    ):
        x.append(feasible_region.project(y - 1 / L * function.grad(y)))
        root = np.roots([1, alpha[-1] ** 2 - q, -alpha[-1] ** 2])
        root = root[(root > 0.0) & (root < 1.0)]
        assert len(root) != 0, "Root does not meet desired criteria.\n"
        alpha.append(root[0])
        beta = alpha[-2] * (1 - alpha[-2]) / (alpha[-2] ** 2 - alpha[-1])
        y = x[-1] + beta * (x[-1] - x[-2])
        grad = function.grad(x[-1])
    return x[-1]


def projected_gradient_descent(
    x0,
    function,
    feasible_region,
    tolerance,
):
    """
    Run projected gradient descent.

    References
    ----------
    Cauchy, A. (1847). Méthode générale pour la résolution des systemes
    d’équations simultanées. Comp. Rend. Sci. Paris, 25(1847), 536-538.

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array.
    feasible_region : feasible region function.
        Returns projection oracle of a point x onto the feasible region,
        which are computed through the function feasible_region.project(x).
        Additionally, a LMO is used to compute the Frank-Wolfe gap (used as a
        stopping criterion) through the function
        feasible_region.linear_optimization_oracle(grad) function, which
        minimizes <x, grad> over the feasible region.
    tolerance : float
        Frank-Wolfe accuracy to which the solution is outputted.

    Returns
    -------
    x : numpy array
        Outputted solution with primal gap below the target tolerance
    """
    x = x0
    grad = function.grad(x)
    L = function.largest_eigenvalue_hessian()
    while grad.dot(x - feasible_region.linear_optimization_oracle(grad)) > tolerance:
        x = feasible_region.project(x - 1 / L * grad)
        grad = function.grad(x)
    return x


def frank_wolfe_step_active_set_update(
    vertex, step_size, maximum_step_size, active_set, active_set_decomposition
):
    if step_size < maximum_step_size:
        flag, index = is_vextex_present(vertex, active_set)
        active_set_decomposition[:] = [
            i * (1 - step_size) for i in active_set_decomposition
        ]
        if flag:
            active_set_decomposition[index] += step_size
        else:
            active_set.append(vertex)
            active_set_decomposition.append(step_size)
    else:
        active_set[:] = [vertex]
        active_set_decomposition[:] = [maximum_step_size]
    return


def away_step_active_set_update(
    vertex_index, step_size, maximum_step_size, active_set, active_set_decomposition
):
    active_set_decomposition[:] = [
        i * (1 + step_size) for i in active_set_decomposition
    ]
    if step_size < maximum_step_size:
        active_set_decomposition[vertex_index] -= step_size
    else:
        del active_set[vertex_index]
        del active_set_decomposition[vertex_index]
    return


def pairwise_step_active_set_update(
    vertex,
    vertex_index,
    step_size,
    maximum_step_size,
    active_set,
    active_set_decomposition,
):
    active_set_decomposition[vertex_index] -= step_size
    if step_size == active_set_decomposition[vertex_index]:
        del active_set[vertex_index]
        del active_set_decomposition[vertex_index]
    # Update the decomposition.
    flag, index = is_vextex_present(vertex, active_set)
    if flag:
        active_set_decomposition[index] += step_size
    else:
        active_set_decomposition.append(step_size)
        active_set.append(vertex.copy())
    return


def frank_wolfe(
    function,
    feasible_region,
    x0=None,
    stopping_criteria=None,
    initial_active_set=None,
    initial_convex_decomposition=None,
    step_size= None,
    algorithm_parameters={
        "algorithm_type": "standard",
        "maintain_active_set": False,
        "lazification_K": 2.0,
        "boosting_minimum_rounds": 2,
        "boosting_delta": 1e-3,
        "return_points": False,
    },
    disable_tqdm=False,
    logging_functions=None,
):
    """
    Run Frank-Wolfe algorithm

    References
    ----------
    Frank, M., & Wolfe, P. (1956). An algorithm for quadratic programming.
    Naval research logistics quarterly, 3(1-2), 95-110.

    Levitin, E. S., & Polyak, B. T. (1966). Constrained minimization methods.
    USSR Computational mathematics and mathematical physics, 6(5), 1-50.

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array.
    feasible_region : feasible region function.
        Returns LP oracles over feasible region. LMOs are computed through a
        feasible_region.linear_optimization_oracle(grad) function that minimizes
        <x, grad> over the feasible region.
    num_it : int
        Number of iterations that will be run of the algorithm.
    step_size_param : dict
        Contains the parameters needed to select a step size procedure.
    return_points : boolean
        Wether to store the history of iterates for the algorithm.
    max_time : float
        Maximum wall-clock time in seconds that the algorithm will be run for.
    frank_wolfe_gap_tolerance : float
        If the algorithm reaches an iterate with a Frank-Wolfe gap below this
        tolerance, it will terminate and output this iterate.

    Returns
    -------
    data : dict
        Dictionary that contains function evaluation and frank-wolfe gap of the
        iterates in data['function_eval'] and data['frank_wolfe_gap']
        respectively. Additionally, it outputs the wall-clock time needed to
        compute the iterates (data['timing']) as well as the iterates
        themselves (optionally, in data['x_val'])
    """
    if "algorithm_type" not in algorithm_parameters:
        algorithm_parameters["algorithm_type"] = "standard"
    assert algorithm_parameters["algorithm_type"] in [
        "lazy",
        "standard",
        "boosted",
        "nep",
    ], "Incorrect algorithm type provided"
    # Set some of the boosting parameters
    if algorithm_parameters["algorithm_type"] == "boosted":
        if "boosting_minimum_rounds" not in algorithm_parameters:
            algorithm_parameters["boosting_minimum_rounds"] = 2
        if "boosting_delta" not in algorithm_parameters:
            algorithm_parameters["boosting_delta"] = 1e-3
    # Set some of the lazy parameters
    if algorithm_parameters["algorithm_type"] == "lazy":
        algorithm_parameters["maintain_active_set"] = True
        if "lazification_K" not in algorithm_parameters:
            algorithm_parameters["lazification_K"] = 2
        grad = function.grad(x0)
        starting_phi_value = grad.dot(
            x0 - feasible_region.linear_optimization_oracle(grad)
        )
        algorithm_parameters["phi_value"] = starting_phi_value
        step_type_to_use = lazy_frank_wolfe_step
    else:
        # This parameter is not needed for the boosted an lazy variants
        starting_phi_value = 0
        step_type_to_use = frank_wolfe_step
    if "maintain_active_set" not in algorithm_parameters:
        algorithm_parameters["maintain_active_set"] = False
    return generic_algorithm_wrapper(
        step_type_to_use,
        function,
        feasible_region,
        x0=x0,
        stopping_criteria=stopping_criteria,
        convex_decomposition=initial_convex_decomposition,
        active_set=initial_active_set,
        step_size= step_size,
        algorithm_parameters=algorithm_parameters,
        disable_tqdm=disable_tqdm,
        logging_functions=logging_functions,
    )


def away_frank_wolfe(
    function,
    feasible_region,
    x0=None,
    stopping_criteria=None,
    initial_active_set=None,
    initial_convex_decomposition=None,
    step_size= None,
    algorithm_parameters={
        "algorithm_type": "standard",
        "lazification_K": 2.0,
        "maintain_active_set": True,
        "return_points": False,
    },
    disable_tqdm=False,
    logging_functions=None,
):
    """
    Run Away-step Frank-Wolfe algorithm

    References
    ----------
    Wolfe, P. (1970). Convergence theory in nonlinear programming. Integer
    and nonlinear programming, 1-36.

    Guélat, J., & Marcotte, P. (1986). Some comments on Wolfe's ‘away step’.
    Mathematical Programming, 35(1), 110-119.

     Braun, G., Pokutta, S., & Zink, D. (2017, July). Lazifying Conditional
     Gradient Algorithms. In ICML (pp. 566-575).

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array.
    feasible_region : feasible region function.
        Returns LP oracles over feasible region. LMOs are computed through a
        feasible_region.linear_optimization_oracle(grad) function that minimizes
        <x, grad> over the feasible region. Away-steps are computed with an
        away oracle, that maximizes <x, grad> over the active set.
    lambda_val : list of floats
        Contains the barycentric coordinates of the input point with respect to
        the active set defined in active_set.
    active_set : list of numpy arrays
        Active set associated with the barycentric coordinates defined above.
    num_it : int
        Number of iterations that will be run of the algorithm.
    step_size_param : dict
        Contains the parameters needed to select a step size procedure.
    return_points : boolean
        Wether to store the history of iterates for the algorithm.
    max_time : float
        Maximum wall-clock time in seconds that the algorithm will be run for.
    frank_wolfe_gap_tolerance : float
        If the algorithm reaches an iterate with a Frank-Wolfe gap below this
        tolerance, it will terminate and output this iterate.

    Returns
    -------
    data : dict
        Dictionary that contains function evaluation and frank-wolfe gap of the
        iterates in data['function_eval'] and data['frank_wolfe_gap']
        respectively. Additionally, it outputs the wall-clock time needed to
        compute the iterates (data['timing']) as well as the iterates
        themselves (optionally, in data['x_val'])
    """
    if "algorithm_type" not in algorithm_parameters:
        algorithm_parameters["algorithm_type"] = "standard"
    assert algorithm_parameters["algorithm_type"] in [
        "lazy",
        "standard",
        "pairwise",
    ], "Incorrect algorithm type provided"
    if algorithm_parameters["algorithm_type"] == "lazy":
        algorithm_parameters["maintain_active_set"] = True
        if "lazification_K" not in algorithm_parameters:
            algorithm_parameters["lazification_K"] = 2
        grad = function.grad(x0)
        starting_phi_value = grad.dot(
            x0 - feasible_region.linear_optimization_oracle(grad)
        )
        algorithm_parameters["phi_value"] = starting_phi_value
        step_type_to_use = lazy_away_frank_wolfe_step
    else:
        step_type_to_use = away_frank_wolfe_step
    return generic_algorithm_wrapper(
        step_type_to_use,
        function,
        feasible_region,
        x0=x0,
        stopping_criteria=stopping_criteria,
        convex_decomposition=initial_convex_decomposition,
        active_set=initial_active_set,
        step_size= step_size,
        algorithm_parameters=algorithm_parameters,
        disable_tqdm=disable_tqdm,
        logging_functions=logging_functions,
    )


def decomposition_invariant_pairwise_frank_wolfe(
    function,
    feasible_region,
    x0=None,
    stopping_criteria=None,
    step_size= None,
    algorithm_parameters={
        "algorithm_type": "standard",
        "return_points": False,
    },
    disable_tqdm=False,
    logging_functions=None,
):
    """
    Run Pairwise Decomposition-invariant Frank-Wolfe algorithm.

    References
    ----------
    Garber, D., & Meshi, O. (2016). Linear-memory and decomposition-invariant
    linearly convergent conditional gradient algorithm for structured
    polytopes. In Advances in neural information processing systems
    (pp. 1001-1009).

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array.
    feasible_region : feasible region function.
        Returns LP oracles over feasible region. LMOs are computed through a
        feasible_region.linear_optimization_oracle(grad) function that minimizes
        <x, grad> over the feasible region.
    lambda_val : list of floats
        Contains the barycentric coordinates of the input point with respect to
        the active set defined in active_set.
    active_set : list of numpy arrays
        Active set associated with the barycentric coordinates defined above.
    num_it : int
        Number of iterations that will be run of the algorithm.
    step_size_param : dict
        Contains the parameters needed to select a step size procedure.
    return_points : boolean
        Wether to store the history of iterates for the algorithm.
    max_time : float
        Maximum wall-clock time in seconds that the algorithm will be run for.
    frank_wolfe_gap_tolerance : float
        If the algorithm reaches an iterate with a Frank-Wolfe gap below this
        tolerance, it will terminate and output this iterate.

    Returns
    -------
    data : dict
        Dictionary that contains function evaluation and frank-wolfe gap of the
        iterates in data['function_eval'] and data['frank_wolfe_gap']
        respectively. Additionally, it outputs the wall-clock time needed to
        compute the iterates (data['timing']) as well as the iterates
        themselves (optionally, in data['x_val'])
    """
    if "algorithm_type" not in algorithm_parameters:
        algorithm_parameters["algorithm_type"] = "standard"
    assert algorithm_parameters["algorithm_type"] in [
        "standard",
        "boosted",
    ], "Incorrect algorithm type provided"
    # Set some of the boosting parameters
    if algorithm_parameters["algorithm_type"] == "boosted":
        if "boosting_minimum_rounds" not in algorithm_parameters:
            algorithm_parameters["boosting_minimum_rounds"] = 2
        if "boosting_delta" not in algorithm_parameters:
            algorithm_parameters["boosting_delta"] = 1e-3
    return generic_algorithm_wrapper(
        decomposition_invariant_pairwise_frank_wolfe_step,
        function,
        feasible_region,
        x0=x0,
        stopping_criteria=stopping_criteria,
        convex_decomposition=None,
        active_set=None,
        step_size= step_size,
        algorithm_parameters=algorithm_parameters,
        disable_tqdm=disable_tqdm,
        logging_functions=logging_functions,
    )


def fully_corrective_frank_wolfe(
    function,
    feasible_region,
    x0=None,
    step_size = None,
    stopping_criteria=None,
    initial_active_set=None,
    initial_convex_decomposition=None,
    algorithm_parameters={
        "inner_stopping_frank_wolfe_gap": 1.0e-6,
        "return_points": False,
    },
    disable_tqdm=False,
    logging_functions=None,
):
    """
    Run Fully-corrective Frank-Wolfe algorithm for the case where the objective
    function is a quadratic function.

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array.
    feasible_region : feasible region function.
        Returns LP oracles over feasible region. LMOs are computed through a
        feasible_region.linear_optimization_oracle(grad) function that minimizes
        <x, grad> over the feasible region.
    lambda_val : list of floats
        Contains the barycentric coordinates of the input point with respect to
        the active set defined in active_set.
    active_set : list of numpy arrays
        Active set associated with the barycentric coordinates defined above.
    num_it : int
        Number of iterations that will be run of the algorithm.
    step_size_param : dict
        Contains the parameters needed to select a step size procedure.
    return_points : boolean
        Wether to store the history of iterates for the algorithm.
    max_time : float
        Maximum wall-clock time in seconds that the algorithm will be run for.
    frank_wolfe_gap_tolerance : float
        If the algorithm reaches an iterate with a Frank-Wolfe gap below this
        tolerance, it will terminate and output this iterate.

    Returns
    -------
    data : dict
        Dictionary that contains function evaluation and frank-wolfe gap of the
        iterates in data['function_eval'] and data['frank_wolfe_gap']
        respectively. Additionally, it outputs the wall-clock time needed to
        compute the iterates (data['timing']) as well as the iterates
        themselves (optionally, in data['x_val'])
    """
    if "inner_stopping_frank_wolfe_gap" not in algorithm_parameters:
        algorithm_parameters["inner_stopping_frank_wolfe_gap"] = 1.0e-6
    return generic_algorithm_wrapper(
        fully_corrective_frank_wolfe_step,
        function,
        feasible_region,
        x0=x0,
        stopping_criteria=stopping_criteria,
        active_set=initial_active_set,
        convex_decomposition=initial_convex_decomposition,
        step_size= step_size,
        algorithm_parameters=algorithm_parameters,
        disable_tqdm=disable_tqdm,
        logging_functions=logging_functions,
    )


def blended_conditional_gradients(
    function,
    feasible_region,
    x0=None,
    stopping_criteria=None,
    initial_active_set=None,
    initial_convex_decomposition=None,
    step_size= None,
    algorithm_parameters={
        "lazification_K": 2.0,
        "return_points": False,
    },
    disable_tqdm=False,
    logging_functions=None,
):
    grad = function.grad(x0)
    starting_phi_value = (
        grad.dot(x0 - feasible_region.linear_optimization_oracle(grad)) / 2.0
    )
    algorithm_parameters["phi_value"] = starting_phi_value
    if "lazification_K" not in algorithm_parameters:
        algorithm_parameters["lazification_K"] = 2
    if (
        "maintain_active_set" in algorithm_parameters
        and algorithm_parameters["maintain_active_set"] == False
    ):
        initial_active_set = []
        initial_convex_decomposition = []
    if "maintain_active_set" not in algorithm_parameters:
        algorithm_parameters["maintain_active_set"] = True
    return generic_algorithm_wrapper(
        blended_conditional_gradients_step,
        function,
        feasible_region,
        x0=x0,
        stopping_criteria=stopping_criteria,
        convex_decomposition=initial_convex_decomposition,
        active_set=initial_active_set,
        step_size= step_size,
        algorithm_parameters=algorithm_parameters,
        disable_tqdm=disable_tqdm,
        logging_functions=logging_functions,
    )


def conditional_gradient_sliding(
    function,
    feasible_region,
    x0=None,
    stopping_criteria=None,
    initial_active_set=None,
    initial_convex_decomposition=None,
    algorithm_parameters={
        "return_points": False,
    },
    disable_tqdm=False,
    logging_functions=None,
):
    """
    Run the Conditional Gradient Sliding algorithm.

    References
    ----------
    Lan, G., & Zhou, Y. (2016). Conditional gradient sliding for convex
    optimization. SIAM Journal on Optimization, 26(2), 1379-1409.

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array. For the finite sum gradient, a stochastic gradient with
        n samples is computed with function.stochastic_grad(x, n).
    feasible_region : feasible region function.
        Returns LP oracles over feasible region. LMOs are computed through a
        feasible_region.linear_optimization_oracle(grad) function that minimizes
        <x, grad> over the feasible region.
    num_it : int
        Number of iterations that will be run of the algorithm.
    step_size_param : dict
        Contains the parameters needed to select a step size procedure.
    return_points : boolean
        Wether to store the history of iterates for the algorithm.
    max_time : float
        Maximum wall-clock time in seconds that the algorithm will be run for.
    frank_wolfe_gap_tolerance : float
        If the algorithm reaches an iterate with a Frank-Wolfe gap below this
        tolerance, it will terminate and output this iterate.

    Returns
    -------
    data : dict
        Dictionary that contains function evaluation and frank-wolfe gap of the
        iterates in data['function_eval'] and data['frank_wolfe_gap']
        respectively. Additionally, it outputs the wall-clock time needed to
        compute the iterates (data['timing']) as well as the iterates
        themselves (optionally, in data['x_val'])
    """
    if function.smallest_eigenvalue_hessian() > 1.0e-3:
        step_type_to_use = conditional_gradient_sliding_str_cvx_step
    else:
        step_type_to_use = conditional_gradient_sliding_cvx_step
    return generic_algorithm_wrapper(
        step_type_to_use,
        function,
        feasible_region,
        x0=x0,
        stopping_criteria=stopping_criteria,
        convex_decomposition=initial_convex_decomposition,
        active_set=initial_active_set,
        step_size= None,
        algorithm_parameters=algorithm_parameters,
        disable_tqdm=disable_tqdm,
        logging_functions=logging_functions,
    )


def locally_accelerated_conditional_gradient_simplex(
    function,
    feasible_region,
    x0=None,
    stopping_criteria=None,
    initial_active_set=None,
    initial_convex_decomposition=None,
    step_size = None,
    algorithm_parameters={
        "return_points": False,
    },
    disable_tqdm=False,
    logging_functions=None,
):
    L = function.largest_eigenvalue_hessian()
    mu = function.smallest_eigenvalue_hessian()
    theta = np.sqrt(0.5 * mu / L)
    dict_iterates = {
        "xAFW": x0.copy(),
        "xAGD": x0.copy(),
        "x": x0.copy(),
        "y": x0.copy(),
        "w": x0.copy(),
        "z": -function.grad(x0) + L * x0,
    }
    auxiliary_dict = {
        "rf": True,
        "rc": 1,
        "A": 1.0,
        "theta": theta,
        "H": int(2 / theta * np.log(0.5 / (theta * theta) - 1)),
    }
    return generic_algorithm_wrapper(
        locally_accelerated_conditional_gradients_step,
        function,
        feasible_region,
        x0=x0,
        stopping_criteria=stopping_criteria,
        convex_decomposition=initial_convex_decomposition,
        active_set=initial_active_set,
        step_size = step_size,
        algorithm_parameters=algorithm_parameters,
        iterates=dict_iterates,
        auxiliary_variables=auxiliary_dict,
        logging_functions=logging_functions,
    )


def frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    convex_decomposition,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    if algorithm_parameters["algorithm_type"] == "boosted":
        d, k, align_g = boosting_procedure(
            feasible_region,
            x,
            grad,
            algorithm_parameters["boosting_delta"],
            algorithm_parameters["boosting_minimum_rounds"],
        )
        data["LMO_calls"].append(data["LMO_calls"][-1] + k)
    elif algorithm_parameters["algorithm_type"] == "nep":
        # Initialize some other oracles for NEP algorithm
        if i == 0:
            data["NEP_calls"] = [0]
        v = feasible_region.nearest_extreme_point_oracle(
            grad, x, L=function.largest_eigenvalue_hessian(), gamma=2 / (2 + i)
        )
        d = v - x
    else:
        d = v - x
    alpha = step_size.compute_step_size(function, x, d, grad, i, maximum_stepsize = 1.0)
    if algorithm_parameters["maintain_active_set"]:
        frank_wolfe_step_active_set_update(
            v,
            alpha,
            1.0,
            active_set,
            convex_decomposition,
        )
    x += alpha * d
    grad = function.grad(x)
    data["FOO_calls"].append(data["FOO_calls"][-1] + 1)
    if algorithm_parameters["algorithm_type"] == "boosted":
        return x, grad, None
    elif algorithm_parameters["algorithm_type"] == "nep":
        data["NEP_calls"].append(data["NEP_calls"][-1] + 1)
        data["LMO_calls"].append(data["LMO_calls"][-1])
        return x, grad, None
    else:
        v = feasible_region.linear_optimization_oracle(grad)
        data["LMO_calls"].append(data["LMO_calls"][-1] + 1)
        return x, grad, v


def away_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    convex_decomposition,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    a, indexMax, lambda_val_max = feasible_region.away_oracle(
        grad, active_set, x, convex_decomposition
    )
    
    if algorithm_parameters["algorithm_type"] == "standard":
        if grad.dot(x - v) > grad.dot(a - x):
            d = v - x
            alpha = step_size.compute_step_size(function, x, d, grad, i, maximum_stepsize = 1.0)
            if algorithm_parameters["maintain_active_set"]:
                frank_wolfe_step_active_set_update(
                    v,
                    alpha,
                    1.0,
                    active_set,
                    convex_decomposition,
                )
        else:
            d = x - a
            alpha = step_size.compute_step_size(function, x, d, grad, i,  maximum_stepsize = lambda_val_max / (1.0 - lambda_val_max))
            if algorithm_parameters["maintain_active_set"]:
                away_step_active_set_update(
                    indexMax,
                    alpha,
                    lambda_val_max / (1.0 - lambda_val_max),
                    active_set,
                    convex_decomposition,
                )
    else:
        d = v - a
        alpha = step_size.compute_step_size(function, x, d, grad, i, maximum_stepsize = lambda_val_max)
        if algorithm_parameters["maintain_active_set"]:
            pairwise_step_active_set_update(
                v,
                indexMax,
                alpha,
                lambda_val_max,
                active_set,
                convex_decomposition,
            )
    x += alpha * d
    grad = function.grad(x)
    v = feasible_region.linear_optimization_oracle(grad)
    # Update number of oracles used
    data["LMO_calls"].append(data["LMO_calls"][-1] + 1)
    data["FOO_calls"].append(data["FOO_calls"][-1] + 1)
    return x, grad, v


def fully_corrective_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    lambda_val,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    probability_simplex_feasible_region = probability_simplex(x.shape[0])
    
    if i == 0:
        data["minimization_convex_hull_oracle"] = [0]
    flag, index = is_vextex_present(v, active_set)
    if not flag:
        active_set.append(v)
        lambda_val.append(0.0)

    # Use PGD
    funcConv = quadratic_function_over_simplex(
        active_set, function.returnM(), function.returnb()
    )
    lambda_val[:] = projected_gradient_descent(
        lambda_val, funcConv, probability_simplex_feasible_region, 1.0e-10
    )
    x = np.zeros(len(active_set[0]))
    for i in range(len(active_set)):
        x += lambda_val[i] * active_set[i]
        
    # new_feasible_region = polytope_defined_by_vertices(active_set)
    # output_results = blended_conditional_gradients(
    #     function,
    #     new_feasible_region,
    #     x0 = new_feasible_region.initial_point(),
    #     stopping_criteria=stopping_criterion(
    #         {
    #             "frank_wolfe_gap": algorithm_parameters[
    #                 "inner_stopping_frank_wolfe_gap"
    #             ],
    #             "timing": 600.0,
    #             "iterations": 1000000,
    #         }
    #     ),
    #     step_size=step_size,
    #     # algorithm_parameters={
    #     #     "algorithm_type": "lazy",
    #     #     "lazification_K": 2.0,
    #     #     "maintain_active_set": False,
    #     #     "return_points": False,
    #     # },
    #     disable_tqdm=True,
    # )
    # x = output_results["solution"]

    grad = function.grad(x)
    v = feasible_region.linear_optimization_oracle(grad)
    # Update number of oracles used
    data["minimization_convex_hull_oracle"].append(
        data["minimization_convex_hull_oracle"][-1] + 1
    )
    data["LMO_calls"].append(data["LMO_calls"][-1] + 1)
    data["FOO_calls"].append(data["FOO_calls"][-1] + 1)
    return x, grad, v


def decomposition_invariant_pairwise_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    lambda_val,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    NegGradAux = np.asarray([-grad[i] if x[i] > 0 else 9e9 for i in range(len(x))])
    a = feasible_region.linear_optimization_oracle(NegGradAux)
    if algorithm_parameters["algorithm_type"] == "boosted":
        d, k, align_g = boosting_procedure(
            feasible_region,
            a,
            grad,
            algorithm_parameters["boosting_delta"],
            algorithm_parameters["boosting_minimum_rounds"],
        )
    else:
        d = v - a
    alpha = step_size.compute_step_size(function, x, d, grad, i, maximum_stepsize = calculate_stepsize_DIPCG(x, d))
    x += alpha * d
    grad = function.grad(x)
    data["FOO_calls"].append(data["FOO_calls"][-1] + 1)
    if algorithm_parameters["algorithm_type"] == "boosted":
        data["LMO_calls"].append(data["LMO_calls"][-1] + k)
        return x, grad, None
    else:
        v = feasible_region.linear_optimization_oracle(grad)
        data["LMO_calls"].append(data["LMO_calls"][-1] + 2)
        return x, grad, v


def lazy_away_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    lambda_val,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    a, indexMax, v, indexMin = max_min_vertex(grad, active_set)
    # Use old FW vertex.
    if (
        grad.dot(x - v) >= grad.dot(a - x)
        and grad.dot(x - v)
        >= algorithm_parameters["phi_value"] / algorithm_parameters["lazification_K"]
    ):
        data["LMO_calls"].append(data["LMO_calls"][-1])
        d = v - x
        alpha = step_size.compute_step_size(function, x, d, grad, i, maximum_stepsize = 1.0)

        if alpha != 1.0:
            lambda_val[:] = [i * (1 - alpha) for i in lambda_val]
            lambda_val[indexMin] += alpha
        else:
            active_set[:] = [v]
            lambda_val[:] = [1.0]

    else:
        if (
            grad.dot(a - x) > grad.dot(x - v)
            and grad.dot(a - x)
            >= algorithm_parameters["phi_value"]
            / algorithm_parameters["lazification_K"]
        ):
            data["LMO_calls"].append(data["LMO_calls"][-1])
            d = x - a
            alpha = step_size.compute_step_size(function, x, d, grad, i, maximum_stepsize = lambda_val[indexMax] / (
                1.0 - lambda_val[indexMax]
            ))
            lambda_val[:] = [i * (1 + alpha) for i in lambda_val]
            # Max step, need to delete a vertex.
            if alpha != lambda_val[indexMax] / (
                1.0 - lambda_val[indexMax]
            ):
                lambda_val[indexMax] -= alpha
            else:
                del active_set[indexMax]
                del lambda_val[indexMax]

        else:
            v = feasible_region.linear_optimization_oracle(grad)
            data["LMO_calls"].append(data["LMO_calls"][-1] + 1)
            if (
                grad.dot(x - v)
                >= algorithm_parameters["phi_value"]
                / algorithm_parameters["lazification_K"]
            ):
                d = v - x
                alpha = step_size.compute_step_size(function, x, d, grad, i, maximum_stepsize = 1.0)

                if alpha != 1.0:
                    lambda_val[:] = [i * (1 - alpha) for i in lambda_val]
                    active_set.append(v)
                    lambda_val.append(alpha)
                else:
                    active_set[:] = [v]
                    lambda_val[:] = [1.0]

            else:
                algorithm_parameters["phi_value"] = min(
                    grad.dot(x - v), algorithm_parameters["phi_value"] / 2.0
                )
                alpha = 0.0
                d = np.zeros(len(x))
    x += alpha * d
    grad = function.grad(x)
    # Update number of oracles used
    data["FOO_calls"].append(data["FOO_calls"][-1] + 1)
    return x, grad, None


def lazy_frank_wolfe_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    lambda_val,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    _, _, v, indexMin = max_min_vertex(grad, active_set)
    if (
        grad.dot(x - v)
        >= algorithm_parameters["phi_value"] / algorithm_parameters["lazification_K"]
    ):
        data["LMO_calls"].append(data["LMO_calls"][-1])
        d = v - x
        alpha = step_size.compute_step_size(function, x, d, grad, i, maximum_stepsize = 1.0)
        if alpha == 1.0:
            active_set[:] = [v]
    else:
        v = feasible_region.linear_optimization_oracle(grad)
        data["LMO_calls"].append(data["LMO_calls"][-1] + 1)
        active_set.append(v)
        if (
            grad.dot(x - v)
            >= algorithm_parameters["phi_value"]
            / algorithm_parameters["lazification_K"]
        ):
            d = v - x
            alpha = step_size.compute_step_size(function, x, d, grad, i,  maximum_stepsize = 1.0)
            if alpha == 1.0:
                active_set[:] = [v]
        else:
            algorithm_parameters["phi_value"] = min(
                grad.dot(x - v), algorithm_parameters["phi_value"] / 2.0
            )
            alpha = 0.0
            d = np.zeros(len(x))
    x += alpha * d
    grad = function.grad(x)
    # Update number of oracles used
    data["FOO_calls"].append(data["FOO_calls"][-1] + 1)
    return x, grad, None


def blended_conditional_gradients_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    lambda_val,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    
    a, _, v, _ = feasible_region.max_min_vertex(grad, active_set, x)
    if grad.dot(a - v) >= algorithm_parameters["phi_value"] and i > 1:
        # print("SIDO ",len(lambda_val), len(active_set), (x>0).sum(), np.abs(x).min())
        x, active_set[:], lambda_val[:] = feasible_region.simplex_descent_oracle(
            function, x, grad, active_set, lambda_val, step_size
        )
        grad = function.grad(x)
        data["LMO_calls"].append(data["LMO_calls"][-1])
        data["FOO_calls"].append(data["FOO_calls"][-1] + 1)
    else:
        val = grad.dot(x - v)
        if (
            val
            < algorithm_parameters["phi_value"] / algorithm_parameters["lazification_K"]
        ):
            v = feasible_region.linear_optimization_oracle(grad)
            data["LMO_calls"].append(data["LMO_calls"][-1] + 1)
            val = grad.dot(x - v)
        else:
            data["LMO_calls"].append(data["LMO_calls"][-1])
        if (
            val
            >= algorithm_parameters["phi_value"]
            / algorithm_parameters["lazification_K"]
        ):
            # print("FW step ",len(lambda_val), len(active_set), (x>0).sum())
            d = v - x
            alpha = step_size.compute_step_size(function, x, d, grad, i)
            if algorithm_parameters["maintain_active_set"]:
                frank_wolfe_step_active_set_update(
                    v, alpha, 1.0, active_set, lambda_val
                )
            x += alpha * d
            grad = function.grad(x)
            data["FOO_calls"].append(data["FOO_calls"][-1] + 1)
        else:
            # print("Adjustment step ",len(lambda_val), len(active_set), (x>0).sum())
            algorithm_parameters["phi_value"] = val / 2.0
            data["FOO_calls"].append(data["FOO_calls"][-1])
    return x, grad, None


def conditional_gradient_sliding_str_cvx_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    lambda_val,
    active_set,
    algorithm_parameters,
    data,
    stopping_criteria,
    **kwargs,
):
    L = function.largest_eigenvalue_hessian()
    mu = function.smallest_eigenvalue_hessian()
    N = int(np.ceil(2 * np.sqrt(6.0 * L / mu)))
    y = x.copy()
    x = x.copy()
    LMO_counter = 0
    for k in range(1, N + 1):
        gamma = 2.0 / (k + 1.0)
        nu = 8.0 * L * data["frank_wolfe_gap"][0] * 2 ** (-i - 1) / (mu * N * k)
        beta = 2.0 * L / k
        z = (1 - gamma) * y + gamma * x
        auxiliary_function = projection_problem_function(x, function.grad(z), beta)
        inner_problem_data = frank_wolfe(
            auxiliary_function,
            feasible_region,
            x0=x,
            stopping_criteria=stopping_criterion(
                {"frank_wolfe_gap": nu, "timing": max(0, stopping_criteria.dict_of_criteria["timing"] -data["timing"][-1])}
            ),
            algorithm_parameters={
                "return_points": False,
            },
            step_size=step_size_class("line_search"),
            disable_tqdm=True,
        )
        if inner_problem_data["timing"][-1] >=  max(0, stopping_criteria.dict_of_criteria["timing"] -data["timing"][-1]):
            data["FOO_calls"].append(data["FOO_calls"][-1])
            data["LMO_calls"].append(data["LMO_calls"][-1])
            return x, None, None
        x = inner_problem_data["solution"]
        y = (1 - gamma) * y + gamma * x
        LMO_counter += len(inner_problem_data["function_eval"]) - 1
    x = y.copy()
    data["FOO_calls"].append(data["FOO_calls"][-1] + N)
    data["LMO_calls"].append(data["LMO_calls"][-1] + LMO_counter)
    return x, None, None


def conditional_gradient_sliding_cvx_step(
    x,
    v,
    grad,
    i,
    step_size,
    function,
    feasible_region,
    lambda_val,
    active_set,
    algorithm_parameters,
    data,
    stopping_criteria,
    **kwargs,
):
    if "y" not in algorithm_parameters:
        algorithm_parameters["y"] = x.copy()
    L = function.largest_eigenvalue_hessian()
    D = feasible_region.diameter()
    gamma = 3.0 / (i + 3.0)
    w = (1 - gamma) * algorithm_parameters["y"] + gamma * x

    auxiliary_function = projection_problem_function(
        x, function.grad(w), 3 * L / (i + 2)
    )
    inner_tolerance = L * D * D / ((i + 1) * (i + 2))
    inner_problem_data = frank_wolfe(
        auxiliary_function,
        feasible_region,
        x0=x,
        stopping_criteria=stopping_criterion(
            {
                "frank_wolfe_gap": inner_tolerance,
                "timing": max(0, stopping_criteria.dict_of_criteria["timing"] -data["timing"][-1]),
            }
        ),
        algorithm_parameters={
            "return_points": False,
        },
        step_size=step_size_class("line_search"),
        disable_tqdm=True,
    )
    x = inner_problem_data["solution"]
    data["FOO_calls"].append(data["FOO_calls"][-1] + 1)
    data["LMO_calls"].append(
        data["LMO_calls"][-1] + len(inner_problem_data["function_eval"]) - 1
    )
    algorithm_parameters["y"] = (1 - gamma) * algorithm_parameters["y"] + gamma * x
    return x, None, None


def locally_accelerated_conditional_gradients_step(
    x,
    v,
    grad,
    i,
    step_size_parameters,
    function,
    feasible_region,
    lambda_val,
    active_set,
    algorithm_parameters,
    data,
    **kwargs,
):
    # Take an accelerated step
    kwargs["iterates"]["xAFW"], vertex_variation = away_step(
        function, feasible_region, x, step_size_parameters
    )
    (
        kwargs["iterates"]["xAGD"],
        kwargs["iterates"]["y"],
        kwargs["iterates"]["z"],
        kwargs["iterates"]["w"],
        kwargs["auxiliary_variables"]["A"],
    ) = accelerated_step(
        function,
        feasible_region,
        kwargs["iterates"]["x"],
        kwargs["iterates"]["y"],
        kwargs["iterates"]["z"],
        kwargs["iterates"]["w"],
        kwargs["auxiliary_variables"]["A"],
        kwargs["auxiliary_variables"]["theta"],
    )
    # Update number of oracles used
    data["LMO_calls"].append(data["LMO_calls"][-1] + 1)
    data["FOO_calls"].append(data["FOO_calls"][-1] + 2)
    if (
        kwargs["auxiliary_variables"]["rf"] == True
        and kwargs["auxiliary_variables"]["rc"] >= kwargs["auxiliary_variables"]["H"]
    ):
        (
            kwargs["iterates"]["xAGD"],
            kwargs["iterates"]["w"],
            kwargs["iterates"]["y"],
            kwargs["iterates"]["z"],
        ) = restart_acceleration(
            function,
            feasible_region,
            kwargs["iterates"]["xAGD"],
            kwargs["iterates"]["xAFW"],
        )
        kwargs["auxiliary_variables"]["A"] = 1
        kwargs["auxiliary_variables"]["rc"] = 0
        kwargs["auxiliary_variables"]["rf"] = False
        # Add extra FOO from restarting
        data["FOO_calls"][-1] += 1
    else:
        (
            kwargs["iterates"]["xAGD"],
            kwargs["iterates"]["y"],
            kwargs["iterates"]["z"],
            kwargs["iterates"]["w"],
            kwargs["auxiliary_variables"]["A"],
        ) = accelerated_step(
            function,
            feasible_region,
            kwargs["iterates"]["x"],
            kwargs["iterates"]["y"],
            kwargs["iterates"]["z"],
            kwargs["iterates"]["w"],
            kwargs["auxiliary_variables"]["A"],
            kwargs["auxiliary_variables"]["theta"],
        )
        # Add extra FOO from accelerated step
        data["FOO_calls"][-1] += 1
        # Keep track of if we have at some point eliminated a vertex.
        if vertex_variation == 1:
            kwargs["auxiliary_variables"]["rf"] = True
    kwargs["auxiliary_variables"]["rc"] += 1
    # Take the best of the two
    if function.f(kwargs["iterates"]["xAGD"]) < function.f(kwargs["iterates"]["xAFW"]):
        kwargs["iterates"]["x"] = kwargs["iterates"]["xAGD"]
    else:
        kwargs["iterates"]["x"] = kwargs["iterates"]["xAFW"]
    return kwargs["iterates"]["x"], None, None


def generic_algorithm_wrapper(
    step_type,
    function,
    feasible_region,
    x0=None,
    stopping_criteria=None,
    convex_decomposition=None,
    active_set=None,
    step_size=None,
    algorithm_parameters={
        "frequency_output": 1,
        "min_initial_output": 1000,
        "maintain_active_set": True,
        "return_points": False,
        "monitor_frank_wolfe_gap": True,
    },
    disable_tqdm=False,
    logging_functions=None,
    **kwargs,
):
    # If no stopping criteria is provided, use a default.
    if stopping_criteria is None:
        stopping_criteria = stopping_criterion(
            {"frank_wolfe_gap": 1.0e-8, "timing": 300.0, "iterations": 1000}
        )

    if step_size is None:
        step_size = step_size_class("adaptive_short_step")

    # Set default parameters if not given
    if "return_points" not in algorithm_parameters:
        algorithm_parameters["return_points"] = False
    if "maintain_active_set" not in algorithm_parameters:
        algorithm_parameters["maintain_active_set"] = True
    if "frequency_output" not in algorithm_parameters:
        algorithm_parameters["frequency_output"] = 1
    if "min_initial_output" not in algorithm_parameters:
        algorithm_parameters["min_initial_output"] = 1
    if "monitor_frank_wolfe_gap" not in algorithm_parameters or (
        stopping_criteria is not None
        and stopping_criteria.dict_of_criteria["frank_wolfe_gap"] > 0
    ):
        algorithm_parameters["monitor_frank_wolfe_gap"] = True

    # If no initial point is provided, use the one provided by the feasible region.
    if x0 is not None:
        x = x0.copy()
    else:
        x = feasible_region.initial_point()
    # Start collecting the data
    start = time.process_time()
    grad = function.grad(x)
    v = feasible_region.linear_optimization_oracle(grad)
    end = time.process_time()
    if algorithm_parameters["maintain_active_set"]:
        if convex_decomposition is not None and active_set is not None:
            convex_decomposition = convex_decomposition.copy()
            active_set = active_set.copy()
        else:
            convex_decomposition = [1.0]
            active_set = [x.copy()]
    data = {
        "function_eval": [function.f(x)],
        "timing": [end - start],
        "LMO_calls": [1],
        "FOO_calls": [1],
    }
    # if algorithm_parameters["monitor_frank_wolfe_gap"]:
    data["frank_wolfe_gap"] = [grad.dot(x - v)]
    if step_size.type_of_step == "adaptive_short_step":
        data["L_estimate"] = [step_size.step_size_parameters["L_estimate"]]
    if algorithm_parameters["return_points"]:
        data["x_val"] = [x]
    if stopping_criteria.evaluate_stopping_criteria(data):
        data["solution"] = x
        return data

    if logging_functions is not None:
        for key, function_value in logging_functions.items():
            data[key] = [function_value(x, active_set, function, feasible_region)]

    # Define the iterable depending on if we have a maximum number of iterations
    # or not.
    if math.isinf(stopping_criteria.dict_of_criteria["iterations"]):
        iterable = itertools.count(start=0)
        total_iterations = np.inf
    else:
        iterable = range(stopping_criteria.dict_of_criteria["iterations"])
        total_iterations = stopping_criteria.dict_of_criteria["iterations"]

    for i in tqdm(iterable, total=total_iterations, disable=disable_tqdm):
        x, grad, v = step_type(
            x,
            v,
            grad,
            i,
            step_size,
            function,
            feasible_region,
            convex_decomposition,
            active_set,
            algorithm_parameters,
            data,
            stopping_criteria = stopping_criteria,
            **kwargs,
        )
        if (
            i % algorithm_parameters["frequency_output"] == 0
            or i <= algorithm_parameters["min_initial_output"]
        ):
            end = time.process_time()
            # In case the steps of the algorithm do not naturally produce the
            # elements needed to compute the FW, compute these, and
            # keep track of the oracle calls.
            if algorithm_parameters["monitor_frank_wolfe_gap"] and grad is None:
                grad = function.grad(x)
                data["FOO_calls"][-1] += 1
            if algorithm_parameters["monitor_frank_wolfe_gap"] and v is None:
                v = feasible_region.linear_optimization_oracle(grad)
                data["LMO_calls"][-1] += 1
            perform_update(
                x,
                v,
                grad,
                function,
                data,
                i,
                end - start,
                return_points=algorithm_parameters["return_points"],
                monitor_frank_wolfe_gap=algorithm_parameters["monitor_frank_wolfe_gap"],
            )
            
            if logging_functions is not None:
                for key, function_value in logging_functions.items():
                    data[key].append(
                        function_value(x, active_set, function, feasible_region)
                    )
            if step_size.type_of_step == "adaptive_short_step":
                data["L_estimate"].append(step_size.step_size_parameters["L_estimate"])
            start = time.process_time()
        if stopping_criteria.evaluate_stopping_criteria(data):
            break
    data["solution"] = x
    return data


def alignment_procedure(d, hat_d):
    if np.linalg.norm(hat_d) < 1e-15:
        return -1
    else:
        return d.dot(hat_d) / (np.linalg.norm(d) * np.linalg.norm(hat_d))


def boosting_procedure(feasible_region, x, grad_f_x, align_tol, K):
    d, Lbd, flag = np.zeros(len(x)), 0, True
    G = grad_f_x + d
    align_d = alignment_procedure(-grad_f_x, d)
    k_tot = K
    for k in range(K):
        u = feasible_region.linear_optimization_oracle(G) - x
        d_norm = np.linalg.norm(d)
        if d_norm > 0 and G.dot(-d / d_norm) < G.dot(u):
            u = -d / d_norm
            flag = False
        lbd = -G.dot(u) / np.linalg.norm(u) ** 2
        dd = d + lbd * u
        align_dd = alignment_procedure(-grad_f_x, dd)
        align_improv = align_dd - align_d
        if align_improv > align_tol:
            d = dd
            Lbd = Lbd + lbd if flag == True else Lbd * (1 - lbd / d_norm)
            G = grad_f_x + d
            align_d = align_dd
            flag = True
        else:
            k_tot = k + 1
            break
    return d / Lbd, k_tot, align_d


def away_step(function, feasible_region, x, step_size):
    start = len(np.where(x > 0.0)[0])
    grad = function.grad(x)
    v = feasible_region.linear_optimization_oracle(grad)
    a, indexMax, lambda_val_max = feasible_region.away_oracle(grad, [], x, [])
    if grad.dot(x - v) > grad.dot(a - x):
        d = v - x
        alpha = step_size.compute_step_size(function, x, d, grad, 0,  maximum_stepsize =1.0)
    else:
        d = x - a
        alpha = step_size.compute_step_size(function, x, d, grad, 0,  maximum_stepsize =lambda_val_max / (1.0 - lambda_val_max))
    x = x + alpha * d
    end = len(np.where(x > 0.0)[0])
    return x, end - start


@jit(nopython=True, cache=True)
def simplex_project(x, alpha):
    n = x.shape[0]
    if x.sum() == alpha:
        for i in range(n):
            if x[i] < 0:
                break
        return x
    v = x - np.max(x)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - alpha)) - 1
    theta = float(cssv[rho] - alpha) / (rho + 1)
    w = np.minimum(alpha, np.maximum(v - theta, 0.0))
    return w


def restart_acceleration(function, feasible_region, xAGD, xAFW):
    if function.f(xAGD) < function.f(xAFW):
        y = xAGD
    else:
        y = xAFW
    z = -function.grad(y) + function.largest_eigenvalue_hessian() * y
    indices = np.where(y > 0.0)[0]
    # Calculate the vector.
    b = z[indices] / function.largest_eigenvalue_hessian()
    # aux = feasible_region.project(b)
    aux = simplex_project(b, feasible_region.alpha)
    w = np.zeros(len(y))
    w[indices] = aux
    return w, w, y, z


def accelerated_step(function, feasible_region, x, y, z, w, A, theta):
    A = A / (1 - theta)
    a = theta * A
    y = (x + theta * w) / (1 + theta)
    z += a * (function.smallest_eigenvalue_hessian() * y - function.grad(y))
    # Compute the projection directly.
    indices = np.where(x > 0.0)[0]
    # Calculate the vector.
    b = z[indices] / (
        function.smallest_eigenvalue_hessian() * A
        + function.largest_eigenvalue_hessian()
        - function.smallest_eigenvalue_hessian()
    )
    # aux = feasible_region.project(b)
    aux = simplex_project(b, feasible_region.alpha)
    w = np.zeros(len(x))
    w[indices] = aux
    return (1 - theta) * x + theta * w, y, z, w, A
