# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 08:50:19 2022

@author: pccom
"""

import numpy as np
from tqdm import tqdm
import time

from frankwolfe.auxiliary_functions import (
    perform_update,
)
from frankwolfe.auxiliary_functions import step_size_class

import tensorflow as tf

from tensorflow.python.ops import (
    math_ops,
    state_ops,
    init_ops,
)
from tensorflow.python.framework import ops


def frank_wolfe_optimal_experiment_design(
    x0,
    function,
    feasible_region,
    algorithm_parameters={
        "maximum_time": 300.0,
        "maximum_iterations": 1000,
        "stopping_frank_wolfe_gap": 1.0e-8,
        "recompute_logdet": False,
    },
    disable_tqdm=False,
):

    if "maximum_iterations" not in algorithm_parameters:
        algorithm_parameters["maximum_iterations"] = 1000
    if "maximum_time" not in algorithm_parameters:
        algorithm_parameters["maximum_time"] = 300.0
    if "stopping_frank_wolfe_gap" not in algorithm_parameters:
        algorithm_parameters["stopping_frank_wolfe_gap"] = 1.0e-8
    if "recompute_logdet" not in algorithm_parameters:
        algorithm_parameters["recompute_logdet"] = False

    x = x0.copy()
    start = time.process_time()
    # Compute inverse matrix
    inv_matrix = np.linalg.inv(function.A.T.dot(np.multiply(x[:, None], function.A)))
    # Compute the gradient
    grad = np.zeros(function.p)
    for i in range(function.p):
        grad[i] = -function.A[i, :].T.dot(inv_matrix).dot(function.A[i, :])
    index_min = np.argmin(grad)
    # Compute the determinant
    determinant = np.linalg.det(function.A.T.dot(np.multiply(x[:, None], function.A)))
    v = feasible_region.linear_optimization_oracle(grad)
    end = time.process_time()
    data = {
        "function_eval": [],
        "frank_wolfe_gap": [grad.dot(x - v)],
        "timing": [end - start],
    }
    if algorithm_parameters["recompute_logdet"]:
        (sign, logdet) = np.linalg.slogdet(
            function.A.T.dot(np.multiply(x[:, None], function.A))
        )
        data["function_eval"].append(-logdet)
    else:
        data["function_eval"].append(-np.log(determinant))
    for i in tqdm(
        range(algorithm_parameters["maximum_iterations"]), disable=disable_tqdm
    ):
        start = time.process_time()
        d = v - x
        alpha = min(
            1.0, (-1 / function.n * grad[index_min] - 1) / (-grad[index_min] - 1)
        )
        # Update the determinant.
        lambda_value = alpha / (1 - alpha)
        determinant = (
            determinant
            * (1 - lambda_value * grad[index_min])
            * (1 + lambda_value) ** (-function.n)
        )
        # Update inverse matrix
        prod1 = inv_matrix.dot(function.A[index_min, :])
        prod2 = function.A[index_min, :].T.dot(inv_matrix)
        inv_matrix = (1 + lambda_value) * (
            inv_matrix
            - lambda_value
            / (1 - lambda_value * grad[index_min])
            * np.outer(prod1, prod2)
        )
        # Update the gradient
        prev_omega = grad[index_min]
        for i in range(function.p):
            if i != index_min:
                grad[i] = -(1 + lambda_value) * (
                    -grad[i]
                    - lambda_value
                    / (1 - lambda_value * prev_omega)
                    * (prod1.dot(function.A[i, :])) ** 2
                )
        grad[index_min] = (
            (1 + lambda_value) * grad[index_min] / (1 - lambda_value * prev_omega)
        )
        index_min = np.argmin(grad)
        # Compute next step

        x += alpha * d
        v = feasible_region.linear_optimization_oracle(grad)
        end = time.process_time()
        if algorithm_parameters["recompute_logdet"]:
            (sign, logdet) = np.linalg.slogdet(
                function.A.T.dot(np.multiply(x[:, None], function.A))
            )
            data["function_eval"].append(-logdet)
        else:
            data["function_eval"].append(-np.log(determinant))
        data["frank_wolfe_gap"].append(grad.dot(x - v))
        data["timing"].append(end - start + data["timing"][-1])
        if data["timing"][-1] > algorithm_parameters["maximum_time"] or (
            data["frank_wolfe_gap"][-1]
            < algorithm_parameters["stopping_frank_wolfe_gap"]
        ):
            break
    data["solution"] = x
    return data


def frank_wolfe_minimum_enclosing_ball(
    x0,
    function,
    feasible_region,
    step_size =step_size_class("line_search",{"alpha_max": 1.0}),
    algorithm_parameters={
        "maximum_time": 300.0,
        "maximum_iterations": 1000,
        "stopping_frank_wolfe_gap": 1.0e-8,
    },
    disable_tqdm=False,
):

    if "maximum_iterations" not in algorithm_parameters:
        algorithm_parameters["maximum_iterations"] = 1000
    if "maximum_time" not in algorithm_parameters:
        algorithm_parameters["maximum_time"] = 300.0
    if "stopping_frank_wolfe_gap" not in algorithm_parameters:
        algorithm_parameters["stopping_frank_wolfe_gap"] = 1.0e-8

    x = x0.copy()
    start = time.process_time()
    grad = function.grad(x)
    v = feasible_region.linear_optimization_oracle(grad)
    end = time.process_time()
    C, radius2, time_radius, cardinality = function.bounding_ball(x)
    data = {
        "function_eval": [function.f(x0)],
        "frank_wolfe_gap": [np.dot(grad, x - v)],
        "timing": [end - start],
        "cardinality": [1],
        "radius2": [0.0],
        "time_radius": [0.0],
    }
    if data["frank_wolfe_gap"][-1] < algorithm_parameters["stopping_frank_wolfe_gap"]:
        data["solution"] = x
        return data
    for i in tqdm(
        range(algorithm_parameters["maximum_iterations"]), disable=disable_tqdm
    ):
        start = time.process_time()
        d = v - x
        alpha = step_size.compute_step_size(function, x, d, grad, i)
        x += alpha * d
        grad = function.grad(x)
        v = feasible_region.linear_optimization_oracle(grad)
        end = time.process_time()
        perform_update(x, v, grad, function, data, i, end - start, return_points=False)
        C, radius2, time_radius, cardinality = function.bounding_ball(x)
        data["cardinality"].append(cardinality)
        data["radius2"].append(radius2)
        data["time_radius"].append(time_radius)
        if data["timing"][-1] > algorithm_parameters["maximum_time"] or (
            data["frank_wolfe_gap"][-1]
            < algorithm_parameters["stopping_frank_wolfe_gap"]
        ):
            break
    data["solution"] = x
    return data


def away_frank_wolfe_minimum_enclosing_ball(
    x0,
    function,
    feasible_region,
    step_size =step_size_class("line_search",{"alpha_max": 1.0}),
    algorithm_parameters={
        "maximum_time": 300.0,
        "maximum_iterations": 1000,
        "stopping_frank_wolfe_gap": 1.0e-8,
    },
    disable_tqdm=False,
):

    if "maximum_iterations" not in algorithm_parameters:
        algorithm_parameters["maximum_iterations"] = 1000
    if "maximum_time" not in algorithm_parameters:
        algorithm_parameters["maximum_time"] = 300.0
    if "stopping_frank_wolfe_gap" not in algorithm_parameters:
        algorithm_parameters["stopping_frank_wolfe_gap"] = 1.0e-8

    x = x0.copy()
    start = time.process_time()
    grad = function.grad(x)
    v = feasible_region.linear_optimization_oracle(grad)
    end = time.process_time()
    C, radius2, time_radius, cardinality = function.bounding_ball(x)
    data = {
        "function_eval": [function.f(x0)],
        "frank_wolfe_gap": [np.dot(grad, x - v)],
        "timing": [end - start],
        "cardinality": [1],
        "radius2": [0.0],
        "time_radius": [0.0],
    }
    if data["frank_wolfe_gap"][-1] < algorithm_parameters["stopping_frank_wolfe_gap"]:
        data["solution"] = x
        return data
    for i in tqdm(
        range(algorithm_parameters["maximum_iterations"]), disable=disable_tqdm
    ):
        start = time.process_time()
        a, indexMax, lambda_val_max = feasible_region.away_oracle(grad, [], x, [])
        if np.dot(grad, x - v) > np.dot(grad, a - x):
            d = v - x
            alpha = step_size.compute_step_size(function, x, d, grad, i, maximum_stepsize = 1.0)
        else:
            d = x - a
            alpha = step_size.compute_step_size(function, x, d, grad, i,  maximum_stepsize = lambda_val_max / (1.0 - lambda_val_max))
        x += alpha * d
        grad = function.grad(x)
        v = feasible_region.linear_optimization_oracle(grad)
        end = time.process_time()
        perform_update(x, v, grad, function, data, i, end - start, return_points=False)
        C, radius2, time_radius, cardinality = function.bounding_ball(x)
        data["cardinality"].append(cardinality)
        data["radius2"].append(radius2)
        data["time_radius"].append(time_radius)
        if data["timing"][-1] > algorithm_parameters["maximum_time"] or (
            data["frank_wolfe_gap"][-1]
            < algorithm_parameters["stopping_frank_wolfe_gap"]
        ):
            break
    data["solution"] = x
    return data


# Away Frank-Wolfe.
def away_frank_wolfe_optimal_experiment_design(
    x0,
    function,
    feasible_region,
    algorithm_parameters={
        "maximum_time": 300.0,
        "maximum_iterations": 1000,
        "stopping_frank_wolfe_gap": 1.0e-8,
        "recompute_logdet": False,
    },
    disable_tqdm=False,
):

    if "maximum_iterations" not in algorithm_parameters:
        algorithm_parameters["maximum_iterations"] = 1000
    if "maximum_time" not in algorithm_parameters:
        algorithm_parameters["maximum_time"] = 300.0
    if "stopping_primal_gap" not in algorithm_parameters:
        algorithm_parameters["stopping_primal_gap"] = 1.0e-8
    if "recompute_logdet" not in algorithm_parameters:
        algorithm_parameters["recompute_logdet"] = False

    x = x0.copy()
    start = time.process_time()

    # Compute inverse matrix
    inv_matrix = np.linalg.inv(function.A.T.dot(np.multiply(x[:, None], function.A)))
    # Compute the gradient
    grad = np.zeros(function.p)
    for i in range(function.p):
        grad[i] = -function.A[i, :].T.dot(inv_matrix).dot(function.A[i, :])
    index_min = np.argmin(grad)
    # Compute the determinant
    determinant = np.linalg.det(function.A.T.dot(np.multiply(x[:, None], function.A)))
    v = feasible_region.linear_optimization_oracle(grad)
    end = time.process_time()
    data = {
        "function_eval": [],
        "frank_wolfe_gap": [grad.dot(x - v)],
        "timing": [end - start],
    }

    if algorithm_parameters["recompute_logdet"]:
        (sign, logdet) = np.linalg.slogdet(
            function.A.T.dot(np.multiply(x[:, None], function.A))
        )
        data["function_eval"].append(-logdet)
    else:
        data["function_eval"].append(-np.log(determinant))
    for i in tqdm(
        range(algorithm_parameters["maximum_iterations"]), disable=disable_tqdm
    ):
        start = time.process_time()
        a, indexMax, lambda_val_max = feasible_region.away_oracle(grad, [], x, [])
        # FW step
        if np.dot(grad, x - v) > np.dot(grad, a - x):
            d = v - x
            index_min = np.argmin(grad)
            alpha = min(
                1.0, (-1 / function.n * grad[index_min] - 1) / (-grad[index_min] - 1)
            )
            # Update the determinant.
            lambda_value = alpha / (1 - alpha)
            determinant = (
                determinant
                * (1 - lambda_value * grad[index_min])
                * (1 + lambda_value) ** (-function.n)
            )
            # Update inverse matrix
            prod1 = inv_matrix.dot(function.A[index_min, :])
            prod2 = function.A[index_min, :].T.dot(inv_matrix)
            inv_matrix = (1 + lambda_value) * (
                inv_matrix
                - lambda_value
                / (1 - lambda_value * grad[index_min])
                * np.outer(prod1, prod2)
            )
            # Update the gradient
            prev_omega = grad[index_min]
            for i in range(function.p):
                if i != index_min:
                    grad[i] = -(1 + lambda_value) * (
                        -grad[i]
                        - lambda_value
                        / (1 - lambda_value * prev_omega)
                        * (prod1.dot(function.A[i, :])) ** 2
                    )
            grad[index_min] = (
                (1 + lambda_value) * grad[index_min] / (1 - lambda_value * prev_omega)
            )
            index_min = np.argmin(grad)
        else:
            # Away step
            d = x - a
            alpha = min(
                lambda_val_max / (1.0 - lambda_val_max),
                (-1 / function.n * grad[indexMax] - 1) / (1 + grad[indexMax]),
            )
            # Update the determinant.
            lambda_value = alpha / (1 - alpha)
            determinant = (
                determinant
                * (1 + alpha / (1 + alpha) * grad[indexMax])
                * (1 + alpha) ** (function.n)
            )
            # Update inverse matrix
            prod1 = inv_matrix.dot(function.A[indexMax, :])
            prod2 = function.A[indexMax, :].T.dot(inv_matrix)
            inv_matrix = (
                1
                / (1 + alpha)
                * (
                    inv_matrix
                    + alpha
                    / (1 + alpha + alpha * grad[indexMax])
                    * np.outer(prod1, prod2)
                )
            )
            # Update the gradient
            prev_omega = grad[indexMax]
            for i in range(function.p):
                grad[i] = (
                    1
                    / (1 + alpha)
                    * (
                        grad[i]
                        - alpha
                        / (1 + alpha + alpha * prev_omega)
                        * (prod1.dot(function.A[i, :])) ** 2
                    )
                )
        x += alpha * d
        v = feasible_region.linear_optimization_oracle(grad)
        end = time.process_time()
        if algorithm_parameters["recompute_logdet"]:
            (sign, logdet) = np.linalg.slogdet(
                function.A.T.dot(np.multiply(x[:, None], function.A))
            )
            data["function_eval"].append(-logdet)
        else:
            data["function_eval"].append(-np.log(determinant))
        data["frank_wolfe_gap"].append(grad.dot(x - v))
        data["timing"].append(end - start + data["timing"][-1])
        if (
            data["timing"][-1] > algorithm_parameters["maximum_time"]
            or data["frank_wolfe_gap"][-1]
            < algorithm_parameters["stopping_frank_wolfe_gap"]
        ):
            break
    data["solution"] = x
    return data


def fully_corrective_frank_wolfe_minimum_enclosing_ball(
    x0,
    function,
    feasible_region,
    true_radius,
    algorithm_parameters={
        "maximum_time": 300.0,
        "maximum_iterations": 1000,
        "stopping_primal_gap": 1.0e-8,
    },
    disable_tqdm=False,
):
    if "maximum_iterations" not in algorithm_parameters:
        algorithm_parameters["maximum_iterations"] = 1000
    if "maximum_time" not in algorithm_parameters:
        algorithm_parameters["maximum_time"] = 300.0
    if "stopping_primal_gap" not in algorithm_parameters:
        algorithm_parameters["stopping_primal_gap"] = 1.0e-8
    x = x0.copy()
    indices = np.where(x > 0.0)[0].tolist()
    center = function.A[:, 0]
    grad = function.grad_center(center)
    v = feasible_region.linear_optimization_oracle(grad)
    data = {
        "function_eval": [function.f(x0)],
        "frank_wolfe_gap": [np.dot(grad, x - v)],
        "timing": [0.0],
        "cardinality": [1],
        "radius2": [0.0],
    }
    for i in tqdm(
        range(algorithm_parameters["maximum_iterations"]), disable=disable_tqdm
    ):
        start = time.process_time()
        grad = function.grad_center(center)
        v = feasible_region.linear_optimization_oracle(grad)
        index = np.where(v > 0.0)[0][0]
        if index not in indices:
            indices.append(index)
        center, radius2, _, cardinality = function.bounding_ball_indices(indices)
        end = time.process_time()
        data["frank_wolfe_gap"].append(np.dot(grad, x - v))
        data["function_eval"].append(radius2)
        data["timing"].append(end - start + data["timing"][-1])
        data["cardinality"].append(cardinality)
        data["radius2"].append(radius2)
        if data["timing"][-1] > algorithm_parameters["maximum_time"] or (
            true_radius + radius2 < algorithm_parameters["stopping_primal_gap"]
        ):
            break
    data["solution"] = x
    return data


def frank_wolfe_online_matrix_completion(
    function,
    feasible_region,
    algorithm_parameters={
        "maximum_time": 1200.0,
        "maximum_iterations": 1000,
    },
    disable_tqdm=False,
):
    if "maximum_iterations" not in algorithm_parameters:
        algorithm_parameters["maximum_iterations"] = 1000
    if "maximum_time" not in algorithm_parameters:
        algorithm_parameters["maximum_time"] = 1200.0

    x = feasible_region.initial_point()
    data = {
        "function_eval": [],
        "timing": [],
    }
    for i in tqdm(
        range(algorithm_parameters["maximum_iterations"]), disable=disable_tqdm
    ):
        # Play x_i and observe the cost function f_i
        f_val, grad = function.play_and_observe(x)
        v = feasible_region.linear_optimization_oracle(grad)
        alpha = min(1.0, 2 / np.sqrt(i + 1))
        d = v - x
        x += alpha * d
        if len(data["function_eval"]) == 0:
            data["function_eval"].append(f_val)
        else:
            data["function_eval"].append(data["function_eval"][-1] + f_val)
            # data["function_eval"].append(f_val)
        data["timing"].append(time.time())
        if (
            data["timing"][-1] - data["timing"][0]
            > algorithm_parameters["maximum_time"]
        ):
            break
    data["timing"][:] = [t - data["timing"][0] for t in data["timing"]]
    return data, x


def gradient_descent_online_matrix_completion(
    function,
    feasible_region,
    algorithm_parameters={
        "maximum_time": 1200.0,
        "maximum_iterations": 1000,
        "D": 1,
        "G": 1,
    },
    disable_tqdm=False,
):
    if "maximum_iterations" not in algorithm_parameters:
        algorithm_parameters["maximum_iterations"] = 1000
    if "maximum_time" not in algorithm_parameters:
        algorithm_parameters["maximum_time"] = 1200.0
    if "G" not in algorithm_parameters:
        algorithm_parameters["G"] = 1
    if "D" not in algorithm_parameters:
        algorithm_parameters["D"] = 1

    x = feasible_region.initial_point()
    data = {
        "function_eval": [],
        "timing": [],
    }
    for i in tqdm(
        range(algorithm_parameters["maximum_iterations"]), disable=disable_tqdm
    ):
        # Play x_i and observe the cost function f_i
        f_val, grad = function.play_and_observe_one_function(x)
        eta = algorithm_parameters["D"] / (algorithm_parameters["G"] * np.sqrt(i + 1))
        x = feasible_region.project(x - eta * grad)
        if len(data["function_eval"]) == 0:
            data["function_eval"].append(f_val)
        else:
            data["function_eval"].append(data["function_eval"][-1] + f_val)
        data["timing"].append(time.time())
        if (
            data["timing"][-1] - data["timing"][0]
            > algorithm_parameters["maximum_time"]
        ):
            break
    data["timing"][:] = [t - data["timing"][0] for t in data["timing"]]
    return data, x


def frank_wolfe_offline_matrix_completion(
    function,
    feasible_region,
    algorithm_parameters={
        "maximum_time": 1200.0,
        "maximum_iterations": 1000,
    },
    disable_tqdm=False,
):
    if "maximum_iterations" not in algorithm_parameters:
        algorithm_parameters["maximum_iterations"] = 1000
    if "maximum_time" not in algorithm_parameters:
        algorithm_parameters["maximum_time"] = 1200.0

    x = feasible_region.initial_point()
    data = {
        "function_eval": [],
        "frank_wolfe_gap": [],
        "timing": [],
    }
    for i in tqdm(
        range(algorithm_parameters["maximum_iterations"]), disable=disable_tqdm
    ):
        grad = function.grad(x)
        v = feasible_region.linear_optimization_oracle(grad)
        d = v - x
        alpha = min(1.0, function.line_search(x, d))
        data["frank_wolfe_gap"].append(np.sum(np.multiply(grad, x - v)))
        x += alpha * d
        if len(data["function_eval"]) == 0:
            data["function_eval"].append(function.f(x))
        else:
            data["function_eval"].append(data["function_eval"][-1] + function.f(x))
        data["timing"].append(time.time())
        if (
            data["timing"][-1] - data["timing"][0]
            > algorithm_parameters["maximum_time"]
        ):
            break
    data["timing"][:] = [t - data["timing"][0] for t in data["timing"]]
    return data, x

import functools
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.training import training_ops
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import context
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import gradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes

def _filter_grads(grads_vars_and_constraints):
    """Filter out iterable with grad equal to None."""
    grads_vars_and_constraints = tuple(grads_vars_and_constraints)
    if not grads_vars_and_constraints:
        return grads_vars_and_constraints
    filtered = []
    vars_with_empty_grads = []
    for gvc in grads_vars_and_constraints:
        grad = gvc[0]
        var = gvc[1]
        if grad is None:
            vars_with_empty_grads.append(var)
        else:
            filtered.append(gvc)
    filtered = tuple(filtered)
    if not filtered:
        raise ValueError("No gradients provided for any variable: %s." %
                         ([v.name for _, v in grads_vars_and_constraints],))
    if vars_with_empty_grads:
        logging.warning(
            "Gradients do not exist for variables %s when minimizing the loss.",
            ([v.name for v in vars_with_empty_grads]))
    return filtered

class ConstrainedOptimizer(tf.keras.optimizers.Optimizer):
    """Base class for Constrained Optimizers"""

    def __init__(self, name='ConstrainedOptimizer', **kwargs):
        super().__init__(name, **kwargs)

    def set_learning_rate(self, learning_rate):
        self._set_hyper("learning_rate", learning_rate)

    def _aggregate_gradients(self, grads_vars_and_constraints):
        """Returns all-reduced gradients.
            Args:
                grads_vars_and_constraints: List of (gradient, variable, constraint) pairs.
            Returns:
                A list of all-reduced gradients.
            """
        grads_and_vars = [(g, v) for g, v, _ in grads_vars_and_constraints]
        filtered_grads_and_vars = _filter_grads(grads_and_vars)

        def all_reduce_fn(distribution, grads_and_vars):
            return distribution.extended.batch_reduce_to(
                ds_reduce_util.ReduceOp.SUM, grads_and_vars)

        if filtered_grads_and_vars:
            reduced = distribute_ctx.get_replica_context().merge_call(
                all_reduce_fn, args=(filtered_grads_and_vars,))
        else:
            reduced = []
        reduced_with_nones = []
        reduced_pos = 0
        for g, _ in grads_and_vars:
            if g is None:
                reduced_with_nones.append(None)
            else:
                reduced_with_nones.append(reduced[reduced_pos])
                reduced_pos += 1
        assert reduced_pos == len(reduced), "Failed to add all gradients"
        return reduced_with_nones

    def _distributed_apply(self, distribution, grads_vars_and_constraints, name, apply_state):
        """`apply_gradients` using a `DistributionStrategy`."""

        def apply_grad_to_update_var(var, grad, constraint):
            """Apply gradient to variable."""
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)

            apply_kwargs = {}
            if isinstance(grad, ops.IndexedSlices):
                if var.constraint is not None:
                    raise RuntimeError(
                        "Cannot use a constraint function on a sparse variable.")
                if "apply_state" in self._sparse_apply_args:
                    apply_kwargs["apply_state"] = apply_state
                return self._resource_apply_sparse_duplicate_indices(
                    grad.values, var, grad.indices, **apply_kwargs)

            if "apply_state" in self._dense_apply_args:
                apply_kwargs["apply_state"] = apply_state
            return self._resource_apply_dense(grad, var, constraint, **apply_kwargs)

        eagerly_outside_functions = ops.executing_eagerly_outside_functions()
        update_ops = []
        with ops.name_scope(name or self._name, skip_on_eager=True):
            for grad, var, constraint in grads_vars_and_constraints:
                def _assume_mirrored(grad):
                    if isinstance(grad, ds_values.PerReplica):
                        return ds_values.Mirrored(grad.values)
                    return grad

                grad = nest.map_structure(_assume_mirrored, grad)
                with distribution.extended.colocate_vars_with(var):
                    with ops.name_scope("update" if eagerly_outside_functions else
                                        "update_" + var.op.name, skip_on_eager=True):
                        update_ops.extend(distribution.extended.update(
                            var, apply_grad_to_update_var, args=(grad, constraint), group=False))

            any_symbolic = any(isinstance(i, ops.Operation) or
                               tf_utils.is_symbolic_tensor(i) for i in update_ops)
            if not context.executing_eagerly() or any_symbolic:
                with ops._get_graph_from_inputs(update_ops).as_default():
                    with ops.control_dependencies(update_ops):
                        return self._iterations.assign_add(1, read_value=False)

            return self._iterations.assign_add(1)

    def apply_gradients(self, grads_vars_and_constraints, name=None, experimental_aggregate_gradients=True):
        grads_vars_and_constraints = _filter_grads(grads_vars_and_constraints)
        var_list = [v for (_, v, _) in grads_vars_and_constraints]
        constraint_list = [c for (_, _, c) in grads_vars_and_constraints]

        with backend.name_scope(self._name):
            with ops.init_scope():
                self._create_all_weights(var_list)

            if not grads_vars_and_constraints:
                return control_flow_ops.no_op()

            if distribute_ctx.in_cross_replica_context():
                raise RuntimeError(
                    "`apply_gradients() cannot be called in cross-replica context. "
                    "Use `tf.distribute.Strategy.run` to enter replica "
                    "context.")

            strategy = distribute_ctx.get_strategy()
            if (not experimental_aggregate_gradients and strategy and isinstance(
                    strategy.extended,
                    parameter_server_strategy.ParameterServerStrategyExtended)):
                raise NotImplementedError(
                    "`experimental_aggregate_gradients=False is not supported for "
                    "ParameterServerStrategy and CentralStorageStrategy")

            apply_state = self._prepare(var_list)
            if experimental_aggregate_gradients:
                reduced_grads = self._aggregate_gradients(grads_vars_and_constraints)
                var_list = [v for _, v, _ in grads_vars_and_constraints]
                grads_vars_and_constraints = list(zip(reduced_grads, var_list, constraint_list))
            return distribute_ctx.get_replica_context().merge_call(
                functools.partial(self._distributed_apply, apply_state=apply_state),
                args=(grads_vars_and_constraints,),
                kwargs={
                    "name": name,
                })


class SFW(ConstrainedOptimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        learning_rate (float): learning rate between 0.0 and 1.0
        momentum (float): momentum factor, 0 for no momentum
        rescale (string or None): Type of learning_rate rescaling. Must be 'diameter', 'gradient' or None
    """

    def __init__(
        self, learning_rate=0.1, momentum=0.9, rescale="diameter", name="SFW", **kwargs
    ):
        super().__init__(name, **kwargs)

        self.rescale = rescale

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("'momentum' must be between [0, 1].")

        self._set_hyper("momentum", kwargs.get("m", momentum))
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

    def _resource_apply_dense(self, grad, var, constraint, apply_state):
        update_ops = []

        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)
        lr = math_ops.cast(self._get_hyper("learning_rate"), var.dtype.base_dtype)

        if self._momentum:
            m = math_ops.cast(self._get_hyper("momentum"), var.dtype.base_dtype)
            momentum_var = self.get_slot(var, "momentum")
            modified_grad = momentum_var.assign(
                math_ops.add(m * momentum_var, (1 - m) * grad)
            )
        else:
            modified_grad = grad

        v = ops.convert_to_tensor(constraint.lmo(modified_grad), var.dtype.base_dtype)
        vminvar = math_ops.subtract(v, var)

        if self.rescale is None:
            factor = math_ops.cast(1.0, var.dtype.base_dtype)
        elif self.rescale == "diameter":
            factor = math_ops.cast(
                1.0 / constraint.get_diameter(), var.dtype.base_dtype
            )
        elif self.rescale == "gradient":
            factor = math_ops.cast(
                tf.norm(modified_grad, ord=2) / tf.norm(vminvar, ord=2),
                var.dtype.base_dtype,
            )
        clipped_lr = math_ops.ClipByValue(
            t=lr * factor, clip_value_min=0, clip_value_max=1
        )

        update_ops.append(state_ops.assign_add(var, clipped_lr * vminvar))

        return control_flow_ops.group(*update_ops)

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum", initializer="zeros")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
            self._get_hyper("momentum", var_dtype)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self._serialize_hyperparameter("momentum"),
            }
        )
        return config


class AdaSFW(ConstrainedOptimizer):
    """AdaGrad Stochastic Frank-Wolfe algorithm.
    Arguments:
        learning_rate (float, optional): learning rate (default: 1e-2)
        inner_steps (integer, optional): number of inner iterations (default: 2)
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
    """

    def __init__(
        self, learning_rate=0.01, inner_steps=2, delta=1e-8, name="AdaSFW", **kwargs
    ):
        super().__init__(name, **kwargs)

        self.K = kwargs.get("K", inner_steps)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("delta", kwargs.get("delta", delta))

    def set_learning_rate(self, learning_rate):
        self._set_hyper("learning_rate", learning_rate)

    def _resource_apply_dense(self, grad, var, constraint, apply_state):
        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)

        learning_rate = math_ops.cast(
            self._get_hyper("learning_rate"), var.dtype.base_dtype
        )
        delta = math_ops.cast(self._get_hyper("delta"), var.dtype.base_dtype)
        accumulator = state_ops.assign_add(
            self.get_slot(var, "accumulator"), math_ops.square(grad)
        )
        H = math_ops.add(delta, math_ops.sqrt(accumulator))
        y = state_ops.assign(self.get_slot(var, "y"), var)

        for idx in range(self.K):
            delta_q = math_ops.add(
                grad,
                math_ops.multiply(
                    H, math_ops.divide(math_ops.subtract(y, var), learning_rate)
                ),
            )
            v = ops.convert_to_tensor(constraint.lmo(delta_q), var.dtype.base_dtype)
            vy_diff = math_ops.subtract(v, y)
            gamma_unclipped = math_ops.divide(
                math_ops.reduce_sum(
                    -learning_rate * math_ops.multiply(delta_q, vy_diff)
                ),
                math_ops.reduce_sum(math_ops.multiply(H, math_ops.square(vy_diff))),
            )
            gamma = math_ops.ClipByValue(
                t=gamma_unclipped, clip_value_min=0, clip_value_max=1
            )
            y = state_ops.assign_add(y, gamma * vy_diff)

        return state_ops.assign(var, y)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(
                var,
                "accumulator",
                init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype),
            )
            self.add_slot(
                var, "y", init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype)
            )

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                learning_rate=self._serialize_hyperparameter("learning_rate"),
                delta=self._serialize_hyperparameter("delta"),
            )
        )
        return config


class AdamSFW(ConstrainedOptimizer):
    def __init__(
        self,
        learning_rate=0.01,
        inner_steps=2,
        delta=1e-8,
        beta1=0.9,
        beta2=0.999,
        name="AdamSFW",
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("delta", kwargs.get("delta", delta))
        self._set_hyper("beta1", kwargs.get("b1", beta1))
        self._set_hyper("beta2", kwargs.get("b2", beta2))
        self.K = kwargs.get("K", inner_steps)

    def set_learning_rate(self, learning_rate):
        self._set_hyper("learning_rate", learning_rate)

    def _resource_apply_dense(self, grad, var, constraint, apply_state):
        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)

        b1 = math_ops.cast(self._get_hyper("beta1"), var.dtype.base_dtype)
        m_accumulator = self.get_slot(var, "m_accumulator")
        m_accumulator.assign(b1 * m_accumulator + (1 - b1) * grad)

        b2 = math_ops.cast(self._get_hyper("beta2"), var.dtype.base_dtype)
        v_accumulator = self.get_slot(var, "v_accumulator")
        v_accumulator.assign(b2 * v_accumulator + (1 - b2) * math_ops.square(grad))

        vhat_accumulator = self.get_slot(var, "vhat_accumulator")
        vhat_accumulator.assign(tf.math.maximum(vhat_accumulator, v_accumulator))

        delta = math_ops.cast(self._get_hyper("delta"), var.dtype.base_dtype)
        H = math_ops.add(delta, math_ops.sqrt(vhat_accumulator))

        learning_rate = math_ops.cast(
            self._get_hyper("learning_rate"), var.dtype.base_dtype
        )

        y = state_ops.assign(self.get_slot(var, "y"), var)

        for idx in range(self.K):
            delta_q = math_ops.add(
                m_accumulator,
                math_ops.multiply(
                    H, math_ops.divide(math_ops.subtract(y, var), learning_rate)
                ),
            )
            v = ops.convert_to_tensor(constraint.lmo(delta_q), var.dtype.base_dtype)
            vy_diff = math_ops.subtract(v, y)
            gamma_unclipped = math_ops.divide(
                math_ops.reduce_sum(
                    -learning_rate * math_ops.multiply(delta_q, vy_diff)
                ),
                math_ops.reduce_sum(math_ops.multiply(H, math_ops.square(vy_diff))),
            )
            gamma = math_ops.ClipByValue(
                t=gamma_unclipped, clip_value_min=0, clip_value_max=1
            )
            y = state_ops.assign_add(y, gamma * vy_diff)

        return state_ops.assign(var, y)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(
                var,
                "m_accumulator",
                init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype),
            )  # , initializer="zeros")
            self.add_slot(
                var,
                "v_accumulator",
                init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype),
            )  # , initializer="zeros")
            self.add_slot(
                var,
                "vhat_accumulator",
                init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype),
            )  # , initializer="zeros")
            self.add_slot(
                var, "y", init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype)
            )  # , initializer="zeros")

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                learning_rate=self._serialize_hyperparameter("learning_rate"),
                delta=self._serialize_hyperparameter("delta"),
                beta1=self._serialize_hyperparameter("beta1"),
                beta2=self._serialize_hyperparameter("beta2"),
            )
        )
        return config


class SGD(ConstrainedOptimizer):
    """Modified SGD which allows projection via Constraint class"""

    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        momentum_style="pytorch_convex",
        weight_decay=0.0,
        name="SGD",
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("weight_decay", kwargs.get("wd", weight_decay))
        self._set_hyper("momentum", kwargs.get("m", momentum))

        self._weight_decay = False
        if (
            isinstance(weight_decay, ops.Tensor)
            or callable(weight_decay)
            or weight_decay > 0
        ):
            self._weight_decay = True

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
            self._momentum_style = momentum_style or "convex_pytorch"
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

    def _resource_apply_dense(self, grad, var, constraint, apply_state):

        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)
        lr = math_ops.cast(self._get_hyper("learning_rate"), var.dtype.base_dtype)

        if self._weight_decay:
            wd = math_ops.cast(self._get_hyper("weight_decay"), var.dtype.base_dtype)
            if self._logging:
                self.get_slot(var, "log.weight_decay").assign_add(wd * lr)
                self.get_slot(var, "log.decoupled_weight_decay").assign_add(wd)
            modified_grad = tf.math.add(grad, wd * var)
        else:
            modified_grad = grad

        if self._momentum:
            m = math_ops.cast(self._get_hyper("momentum"), var.dtype.base_dtype)
            momentum_var = self.get_slot(var, "momentum")

            if "original" in self._momentum_style:
                if "convex" in self._momentum_style:
                    momentum_var.assign(
                        math_ops.add(m * momentum_var, (1 - m) * lr * modified_grad)
                    )
                else:
                    momentum_var.assign(
                        math_ops.add(m * momentum_var, lr * modified_grad)
                    )
                var_update = state_ops.assign_sub(var, momentum_var)
            elif "pytorch" in self._momentum_style:
                if "convex" in self._momentum_style:
                    momentum_var.assign(
                        math_ops.add(m * momentum_var, (1 - m) * modified_grad)
                    )
                else:
                    momentum_var.assign(math_ops.add(m * momentum_var, modified_grad))
                var_update = state_ops.assign_sub(var, lr * momentum_var)
            else:
                raise NotImplementedError(
                    f"Unknown momentum style {self._momentum_style}"
                )
        else:
            var_update = state_ops.assign_sub(var, lr * modified_grad)

        project_var, was_projected = constraint.euclidean_project(var)
        return tf.cond(
            was_projected,
            true_fn=lambda: state_ops.assign(var, project_var),
            false_fn=lambda: var_update,
        )

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum", initializer="zeros")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
            self._get_hyper("momentum", var_dtype)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "momentum": self._serialize_hyperparameter("momentum"),
            }
        )
        return config


class Adam(ConstrainedOptimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        name="Adam",
        **kwargs,
    ):
        super(Adam, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Adam, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper("beta_2", var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = apply_state[(var_device, var_dtype)]["lr_t"] * (
            math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        )
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
            )
        )

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super(Adam, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, constraint, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        if not self.amsgrad:
            var_update = training_ops.resource_apply_adam(
                var.handle,
                m.handle,
                v.handle,
                coefficients["beta_1_power"],
                coefficients["beta_2_power"],
                coefficients["lr_t"],
                coefficients["beta_1_t"],
                coefficients["beta_2_t"],
                coefficients["epsilon"],
                grad,
                use_locking=self._use_locking,
            )
        else:
            vhat = self.get_slot(var, "vhat")
            var_update = training_ops.resource_apply_adam_with_amsgrad(
                var.handle,
                m.handle,
                v.handle,
                vhat.handle,
                coefficients["beta_1_power"],
                coefficients["beta_2_power"],
                coefficients["lr_t"],
                coefficients["beta_1_t"],
                coefficients["beta_2_t"],
                coefficients["epsilon"],
                grad,
                use_locking=self._use_locking,
            )

        project_var, was_projected = constraint.euclidean_project(var)
        return state_ops.assign(var, project_var)

    def get_config(self):
        config = super(Adam, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._serialize_hyperparameter("decay"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


class Adadelta(ConstrainedOptimizer):
    def __init__(
        self, learning_rate=0.001, rho=0.95, epsilon=1e-7, name="Adadelta", **kwargs
    ):
        super(Adadelta, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("rho", rho)
        self.epsilon = epsilon or backend_config.epsilon()

    def _create_slots(self, var_list):
        # Separate for-loops to respect the ordering of slot variables from v1.
        for v in var_list:
            self.add_slot(v, "accum_grad")
        for v in var_list:
            self.add_slot(v, "accum_var")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Adadelta, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)].update(
            dict(
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                rho=array_ops.identity(self._get_hyper("rho", var_dtype)),
            )
        )

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras V1 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super(Adadelta, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, constraint, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        accum_grad = self.get_slot(var, "accum_grad")
        accum_var = self.get_slot(var, "accum_var")
        var_update = training_ops.resource_apply_adadelta(
            var.handle,
            accum_grad.handle,
            accum_var.handle,
            coefficients["lr_t"],
            coefficients["rho"],
            coefficients["epsilon"],
            grad,
            use_locking=self._use_locking,
        )

        project_var, was_projected = constraint.euclidean_project(var)
        return state_ops.assign(var, project_var)

    def get_config(self):
        config = super(Adadelta, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._serialize_hyperparameter("decay"),
                "rho": self._serialize_hyperparameter("rho"),
                "epsilon": self.epsilon,
            }
        )
        return config


class Adagrad(ConstrainedOptimizer):
    def __init__(self, learning_rate=0.001, delta=1e-8, name="Adagrad", **kwargs):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("delta", kwargs.get("delta", delta))

    def set_learning_rate(self, learning_rate):
        self._set_hyper("learning_rate", learning_rate)

    def _resource_apply_dense(self, grad, var, constraint, apply_state=None):
        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)

        learning_rate = math_ops.cast(
            self._get_hyper("learning_rate"), var.dtype.base_dtype
        )
        delta = math_ops.cast(self._get_hyper("delta"), var.dtype.base_dtype)
        accumulator = state_ops.assign_add(
            self.get_slot(var, "accumulator"), math_ops.square(grad)
        )
        H = math_ops.add(delta, math_ops.sqrt(accumulator))
        sqrtH = math_ops.sqrt(H)

        project_var = constraint.qmo(
            sqrtH * var - learning_rate * grad / sqrtH, 1 / sqrtH
        )
        return state_ops.assign(var, project_var / sqrtH)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(
                var,
                "accumulator",
                init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype),
            )

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                learning_rate=self._serialize_hyperparameter("learning_rate"),
                delta=self._serialize_hyperparameter("delta"),
            )
        )
        return config



class AMSGrad(ConstrainedOptimizer):
    def __init__(
        self,
        learning_rate=0.01,
        delta=1e-8,
        beta1=0.9,
        beta2=0.999,
        name="AMSGrad",
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("delta", kwargs.get("delta", delta))
        self._set_hyper("beta1", kwargs.get("b1", beta1))
        self._set_hyper("beta2", kwargs.get("b2", beta2))

    def set_learning_rate(self, learning_rate):
        self._set_hyper("learning_rate", learning_rate)

    def _resource_apply_dense(self, grad, var, constraint, apply_state):
        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)

        b1 = math_ops.cast(self._get_hyper("beta1"), var.dtype.base_dtype)
        m_accumulator = self.get_slot(var, "m_accumulator")
        momentum = m_accumulator.assign(b1 * m_accumulator + (1 - b1) * grad)

        b2 = math_ops.cast(self._get_hyper("beta2"), var.dtype.base_dtype)
        v_accumulator = self.get_slot(var, "v_accumulator")
        v_accumulator.assign(b2 * v_accumulator + (1 - b2) * math_ops.square(grad))

        vhat_accumulator = self.get_slot(var, "vhat_accumulator")
        vhat_accumulator.assign(tf.math.maximum(vhat_accumulator, v_accumulator))

        delta = math_ops.cast(self._get_hyper("delta"), var.dtype.base_dtype)
        H = math_ops.add(delta, math_ops.sqrt(vhat_accumulator))
        sqrtH = math_ops.sqrt(H)

        learning_rate = math_ops.cast(
            self._get_hyper("learning_rate"), var.dtype.base_dtype
        )

        project_var = constraint.qmo(
            sqrtH * var - learning_rate * momentum / sqrtH, 1 / sqrtH
        )
        return state_ops.assign(var, project_var / sqrtH)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(
                var,
                "m_accumulator",
                init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype),
            )
            self.add_slot(
                var,
                "v_accumulator",
                init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype),
            )
            self.add_slot(
                var,
                "vhat_accumulator",
                init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype),
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                learning_rate=self._serialize_hyperparameter("learning_rate"),
                delta=self._serialize_hyperparameter("delta"),
                beta1=self._serialize_hyperparameter("beta1"),
                beta2=self._serialize_hyperparameter("beta2"),
            )
        )
        return config
