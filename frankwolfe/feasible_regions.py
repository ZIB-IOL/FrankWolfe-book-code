import numpy as np
from .auxiliary_functions import (
    max_min_vertex,
    calculate_stepsize_SIDO,
    brute_force_away_oracle,
)

"""LP model based on Gurobi solver."""
from gurobipy import GRB, read, Column

import random
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment


from abc import ABC, abstractmethod


def projection_simplex_sort(vect, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = vect.shape  # will raise ValueError if v is not 1-D
    if vect.sum() == s and np.alltrue(vect >= 0):
        return vect
    v = vect - np.max(vect)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - s)) - 1
    theta = float(cssv[rho] - s) / (rho + 1)
    w = (v - theta).clip(min=0)
    return w


def simplex_descent_oracle(
    function,
    x,
    gradient,
    active_set,
    barycentric_coordinates,
    step_size=None,
):
    num_vertex_active_set = len(active_set)
    active_set_matrix = np.vstack(active_set)
    g = active_set_matrix.dot(gradient)
    d = g - g.dot(np.ones(num_vertex_active_set)) / num_vertex_active_set
    lambda_val = np.asarray(barycentric_coordinates)
    if not d.any():
        return active_set[0], [active_set[0]], [1.0]
    else:
        gamma, index = calculate_stepsize_SIDO(lambda_val, d)
        y = x - gamma * d.dot(active_set_matrix)
        if function.f(y) <= function.f(x):
            # print("Deleting", (y>0).sum(), np.abs(y).min(), sorted(y))
            barycentric_coordinates = list(lambda_val - gamma * d)
            del active_set[index]
            del barycentric_coordinates[index]
            return y, active_set, barycentric_coordinates
        else:
            alpha = step_size.compute_step_size(function, x, y - x, gradient, 0,maximum_stepsize = 1 )
            return x + alpha * (y - x), active_set, list(lambda_val - alpha * gamma * d)


class _FeasibleRegion(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def initial_point(self):
        raise NotImplementedError(
            "The feasible region does not have a method to output an initial point"
        )

    @abstractmethod
    def linear_optimization_oracle(self, gradient):
        pass

    def away_oracle(self, grad, active_set, x, lambda_vals):
        """
        Define a brute-force away oracle.
        Parameters
        ----------
        grad: np.ndarray
            Gradient for the away oracle.
        active_set: list of numpy arrays
            Support of the current point.
        x: np.ndarray
            Point x at which the grad is computed
        lambda_vals: list of floats
            Barycentric coordinates for the current point
        Returns
        -------
        Numpy array
        """
        vertex, index_max = brute_force_away_oracle(grad, active_set)
        return vertex, index_max, lambda_vals[index_max]

    def max_min_vertex(self, grad, active_set, x):
        """
        Returns the points in the active set which have greatest/smallest inner product
        with the gradient, as well as the location of these points in the
        barycentric representation
        Parameters
        ----------
        grad: np.ndarray
            Gradient for the away oracle.
        active_set: list of numpy arrays
            Support of the current point.
        x: np.ndarray
            Point x at which the grad is computed
        Returns
        -------
        Numpy array of greatest inner product, index representing the location
        of aforementioned point, Numpy array of smallest inner product, index
        representing the location of aforementioned point
        """
        return max_min_vertex(grad, active_set)

    def simplex_descent_oracle(
        self, function, x, grad, active_set, lambda_val, step_size_param
    ):
        return simplex_descent_oracle(
            function, x, grad, active_set, lambda_val, step_size_param
        )

    def project(self, x):
        """
        Perform a Euclidean projection of x onto the feasible region.
        Parameters
        ----------
        x: np.ndarray
            Point x at which the grad is computed
        Returns
        -------
        Numpy array
        """
        raise NotImplementedError(
            "The feasible region does not have a method to perform a projection"
        )


class polytope_defined_by_vertices(_FeasibleRegion):
    def __init__(self, vertices):
        self.vertices = vertices.copy()
        return

    def initial_point(self):
        return self.vertices[0]

    def linear_optimization_oracle(self, grad):
        vertex, index_max = brute_force_away_oracle(-grad, self.vertices)
        return vertex


class L1_ball(_FeasibleRegion):
    def __init__(self, dimension, alpha=1.0):
        self.len = dimension
        self.alpha = alpha
        return

    def initial_point(self):
        v = np.zeros(self.len, dtype=float)
        v[0] = self.alpha
        return v

    def linear_optimization_oracle(self, grad):
        v = np.zeros(len(grad), dtype=float)
        maxInd = np.argmax(np.abs(grad))
        v[maxInd] = -self.alpha * np.sign(grad[maxInd])
        return v

    def project(self, v):
        u = np.abs(v)
        if u.sum() <= self.alpha:
            return v
        w = projection_simplex_sort(u, s=self.alpha)
        w *= np.sign(v)
        return w

    def diameter(self):
        return 2.0 * self.alpha


class L2_ball(_FeasibleRegion):
    def __init__(self, dimension, alpha=1.0):
        self.len = dimension
        self.alpha = alpha
        return

    def initial_point(self):
        v = np.ones(self.len, dtype=float)
        return v * self.alpha / np.linalg.norm(v)

    def linear_optimization_oracle(self, grad):
        v = grad.copy()
        return -v * self.alpha / np.linalg.norm(grad)

    def project(self, v):
        if np.linalg.norm(v) < self.alpha:
            return v
        else:
            return v * self.alpha / np.linalg.norm(v)

    def diameter(self):
        return np.sqrt(2.0) * self.alpha


class probability_simplex(_FeasibleRegion):
    def __init__(self, dimension, alpha=1.0):
        self.len = dimension
        self.alpha = alpha
        self.initial_point_index = random.randint(0, dimension - 1)
        return

    def initial_point(self):
        v = np.zeros(self.len, dtype=float)
        v[self.initial_point_index] = self.alpha
        return v

    def linear_optimization_oracle(self, grad):
        v = np.zeros(len(grad), dtype=float)
        v[np.argmin(grad)] = self.alpha
        return v

    def project(self, v):
        return projection_simplex_sort(v, s=self.alpha)

    def diameter(self):
        return np.sqrt(2.0) * self.alpha

    def nearest_extreme_point_oracle(self, grad, x, L=0.0, gamma=0.0):
        return self.linear_optimization_oracle(grad - L * gamma * x)

    def away_oracle(self, grad, active_set, x, lambda_vals):
        aux = np.multiply(grad, np.sign(x))
        indices = np.where(x > 0.0)[0]
        v = np.zeros(len(x), dtype=float)
        max_index_aux = np.argmax(aux[indices])
        index_max = indices[max_index_aux]
        v[index_max] = self.alpha
        return v, max_index_aux, x[index_max] / self.alpha

    def max_min_vertex(self, grad, active_set, x):
        aux = np.multiply(grad, np.sign(x))
        indices = np.where(x > 0.0)[0]
        v = np.zeros(len(x), dtype=float)
        u = np.zeros(len(x), dtype=float)
        index_max = indices[np.argmax(aux[indices])]
        index_min = indices[np.argmin(aux[indices])]
        v[index_max] = self.alpha
        u[index_min] = self.alpha
        return v, index_max, u, index_min

    def simplex_descent_oracle(
        self,
        function,
        x,
        grad,
        active_set,
        lambda_val,
        step_size_param,
    ):
        x, active_set[:], lambda_val[:] = simplex_descent_oracle(
            function, x, grad, active_set, lambda_val, step_size_param
        )
        x[np.abs(x) < 1.0e-12] = 0
        return x, active_set[:], lambda_val[:]

class birkhoff_polytope(_FeasibleRegion):
    def __init__(self, dim):
        self.dim = dim
        self.matdim = int(np.sqrt(dim))
        return

    def initial_point(self):
        return np.identity(self.matdim).flatten()

    def linear_optimization_oracle(self, grad):
        objective = grad.reshape((self.matdim, self.matdim))
        matching = linear_sum_assignment(objective)
        solution = np.zeros((self.matdim, self.matdim))
        solution[matching] = 1
        return solution.reshape(self.dim)

    def diameter(self):
        return np.sqrt(2.0 * self.matdim)

    def nearest_extreme_point_oracle(self, grad, x, L=0.0, gamma=0.0):
        return self.linear_optimization_oracle(grad - L * gamma * x)


class box_constraints(_FeasibleRegion):
    def __init__(self, lower_bounds, upper_bounds):
        self.upper_bounds = np.asarray(upper_bounds)
        self.lower_bounds = np.asarray(lower_bounds)
        assert np.all(
            self.upper_bounds >= self.lower_bounds
        ), "Upper and lower bound constraints are incorrect."
        return

    def initial_point(self):
        return self.upper_bounds

    def linear_optimization_oracle(self, grad):
        assert len(grad) == len(
            self.upper_bounds
        ), "Size of gradient does not match that of upper/lower bound"
        v = np.zeros(len(grad), dtype=float)
        upper = np.multiply(grad, self.upper_bounds)
        lower = np.multiply(grad, self.lower_bounds)
        for i in range(len(grad)):
            if upper[i] <= lower[i]:
                v[i] = self.upper_bounds[i]
            else:
                v[i] = self.lower_bounds[i]
        return v

    def diameter(self):
        return np.linalg.norm(self.upper_bounds - self.lower_bounds)

    def nearest_extreme_point_oracle(self, grad, x, L=0.0, gamma=0.0):
        assert np.all(self.lower_bounds == 0.0) and np.all(
            self.upper_bounds == 1.0
        ), "Incorrect bounds for the NEP"
        return self.linear_optimization_oracle(grad - L * gamma / 2 * (2 * x - 1))


class nuclear_norm_ball(_FeasibleRegion):
    def __init__(self, dim1, dim2, alpha=1.0, flatten=False):
        self.dim1 = dim1
        self.dim2 = dim2
        self.alpha = alpha
        self.flatten = flatten
        return

    def initial_point(self):
        return (np.identity(self.dim1) / self.dim1).flatten()
        # return self.linear_optimization_oracle(np.ones(int(self.dim1 * self.dim2)))

    def linear_optimization_oracle(self, X):
        objective = X.reshape((self.dim1, self.dim2))
        u, s, vt = svds(-objective, k=1, which="LM")
        if self.flatten:
            return (self.alpha * np.outer(u.flatten(), vt.flatten())).flatten()
        else:
            return self.alpha * np.outer(u.flatten(), vt.flatten())

    def project(self, X):
        """Projection onto nuclear norm ball."""
        U, s, V = np.linalg.svd(X, full_matrices=False)
        s = projection_simplex_sort(s)
        if self.flatten:
            return (self.alpha * U.dot(np.diag(s).dot(V))).flatten()
        else:
            return self.alpha * U.dot(np.diag(s).dot(V))

    def diameter(self):
        return np.sqrt(2)*self.alpha

class flow_polytope(_FeasibleRegion):
    def __init__(self, number_nodes_per_layer, number_of_layers):
        self.number_nodes_per_layer = number_nodes_per_layer
        self.number_of_layers = number_of_layers
        I = np.identity(20)
        self.lmo = lambda g: np.concatenate(
            [
                I[
                    np.argmin(
                        g[
                            self.number_nodes_per_layer
                            * i : self.number_nodes_per_layer
                            * i
                            + self.number_nodes_per_layer
                        ]
                    )
                ]
                for i in range(self.number_of_layers)
            ],
            axis=0,
        )
        self.dim = self.number_nodes_per_layer * self.number_of_layers
        return

    def linear_optimization_oracle(self, x):
        return self.lmo(x)

    def initial_point(self):
        aux = np.random.rand(self.dim)
        return self.linear_optimization_oracle(aux)


class gurobi_polytope(_FeasibleRegion):
    def __init__(
        self, modelFilename, addCubeConstraints=False, transform_to_equality=False
    ):
        self.run_config = {
            "solution_only": True,
            "verbosity": "normal",
            "OutputFlag": 0,
            "dual_gap_acc": 1e-06,
            "runningTimeLimit": None,
            "use_LPSep_oracle": True,
            "max_lsFW": 100000,
            "strict_dropSteps": True,
            "max_stepsSub": 100000,
            "max_lsSub": 100000,
            "LPsolver_timelimit": 100000,
            "K": 1,
        }

        model = read(modelFilename)
        model.setParam("OutputFlag", False)
        model.params.TimeLimit = self.run_config["LPsolver_timelimit"]
        model.params.threads = 4
        model.params.MIPFocus = 0
        model.update()
        if addCubeConstraints:
            counter = 0
            for v in model.getVars():
                model.addConstr(v <= 1, "unitCubeConst" + str(counter))
                counter += 1
        model.update()
        if transform_to_equality:
            for c in model.getConstrs():
                sense = c.sense
                if sense == GRB.GREATER_EQUAL:
                    model.addVar(
                        obj=0, name="ArtN_" + c.constrName, column=Column([-1], [c])
                    )
                if sense == GRB.LESS_EQUAL:
                    model.addVar(
                        obj=0, name="ArtP_" + c.constrName, column=Column([1], [c])
                    )
                c.sense = GRB.EQUAL
        model.update()
        self.dimension = len(model.getVars())
        self.model = model
        return

    """
    To find the total number of constraints in a model: model.NumConstrs
    To return the constraints of a model: model.getConstrs()
    To add a single constraint to the model model.addConstr(model.getVars()[-1] == 0, name = 'newConstraint1')
    If we want to delete the last constraint that was added we do: model.remove(model.getConstrs()[-1])
    """

    def linear_optimization_oracle(self, cc):
        """Find good solution for cc with optimality callback."""
        m = self.model
        for it, v in enumerate(m.getVars()):
            v.setAttr(GRB.attr.Obj, cc[it])
        # Update the model with the new atributes.
        m.update()
        m.optimize(lambda mod, where: self.fakeCallback(mod, where, GRB.INFINITY))
        # Status checking
        status = m.getAttr(GRB.Attr.Status)
        if (
            status == GRB.INF_OR_UNBD
            or status == GRB.INFEASIBLE
            or status == GRB.UNBOUNDED
        ):
            assert (
                False
            ), "The model cannot be solved because it is infeasible or unbounded"
        if status != GRB.OPTIMAL:
            print(status)
            assert False, "Optimization was stopped."
        # Store the solution that will be outputted.
        solution = np.array([v.x for v in m.getVars()], dtype=float)[:]
        # Check that the initial number of constraints and the final number is the same.
        return solution

    def initial_point(self):
        return self.linear_optimization_oracle(np.zeros(self.dimension))

    def dim(self):
        return self.dimension

    def fakeCallback(self, model, where, value):
        ggEps = 1e-08
        if where == GRB.Callback.MIPSOL:
            # x = model.cbGetSolution(model.getVars())
            # logging.info 'type of x: ' + str(type(x))
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < value - ggEps:
                print("early termination with objective value :{}".format(obj))
                print("which is better than {}".format(value - ggEps))
                # model.terminate()

        if where == GRB.Callback.MIP:
            objBnd = model.cbGet(GRB.Callback.MIP_OBJBND)

            if objBnd >= value + ggEps:
                # model.terminate()
                pass


# ##### LMO CLASSES FOR NONCONVEX STOCHASTIC EXPERIMENTS WITH TENSORFLOW


import tensorflow as tf

def get_avg_init_norm(shape, initializer="glorot_uniform", ord=2, repetitions=100):
    """Computes the average norm of default layer initialization"""
    initializer = getattr(tf.keras.initializers, initializer)()
    return np.mean(
        [tf.norm(initializer(shape), ord=2).numpy() for _ in range(repetitions)]
    )


def convert_lp_radius(radius, n, in_ord=2, out_ord=np.inf):
    """
    Convert between radius of Lp balls such that the ball of order out_order
    has the same L2 diameter as the ball with radius r of order in_order
    in N dimensions
    """
    in_ord_rec = 0.5 if in_ord == 1 else 1.0 / in_ord
    out_ord_rec = 0.5 if out_ord == 1 else 1.0 / out_ord
    return radius * n ** (out_ord_rec - in_ord_rec)


def create_lInf_constraints(
    model, value=300, mode="initialization", initializer="glorot_uniform"
):
    constraints = []

    for var in model.trainable_variables:
        n = tf.size(var).numpy()

        if mode == "radius":
            constraint = LInfBall(n, diameter=None, radius=value)
        elif mode == "diameter":
            constraint = LInfBall(n, diameter=value, radius=None)
        elif mode == "initialization":
            avg_norm = get_avg_init_norm(
                var.shape, initializer=initializer, ord=float("inf")
            )
            diameter = 2.0 * value * avg_norm
            constraint = LInfBall(n, diameter=diameter, radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")

        constraints.append(constraint)

    return constraints


class Constraint:
    """
    Parent/Base class for constraints
    :param n: dimension of constraint parameter space
    """

    def __init__(self, n):
        self.n = n
        self._diameter, self._radius = None, None

    def get_diameter(self):
        return self._diameter

    def get_radius(self):
        try:
            return self._radius
        except:
            raise ValueError("Tried to get radius from a constraint without one")

    def lmo(self, x):
        assert (
            np.prod(x.shape) == self.n
        ), f"shape {x.shape} does not match dimension {self.n}"

    def qmo(self, x, a):
        assert (
            np.prod(x.shape) == self.n
        ), f"shape {x.shape} does not match dimension {self.n}"
        assert (
            np.prod(a.shape) == self.n
        ), f"shape {a.shape} of accumulator does not match dimension {self.n}"

    def shift_inside(self, x):
        assert (
            np.prod(x.shape) == self.n
        ), f"shape {x.shape} does not match dimension {self.n}"

    def euclidean_project(self, x):
        assert (
            np.prod(x.shape) == self.n
        ), f"shape {x.shape} does not match dimension {self.n}"


# #### LMO CLASSES ####
class LInfBall(Constraint):
    """
    LMO class for the n-dim Lp-Ball (p=ord) with L2-diameter diameter or radius.
    """

    def __init__(self, n, diameter=None, radius=None):
        super().__init__(n)

        self.p = np.inf
        self.q = 1

        if diameter is None and radius is None:
            raise ValueError("Neither diameter and radius given")
        elif diameter is None:
            self._radius = radius
            self._diameter = 2 * convert_lp_radius(
                radius, self.n, in_ord=self.p, out_ord=2
            )
        elif radius is None:
            self._radius = convert_lp_radius(
                diameter / 2.0, self.n, in_ord=2, out_ord=self.p
            )
            self._diameter = diameter
        else:
            raise ValueError("Both diameter and radius given")

    def lmo(self, x):
        """Returns v with norm(v, self.p) <= r minimizing v*x"""
        super().lmo(x)
        random_part = self._radius * tf.cast(
            2 * tf.random.uniform(x.shape, minval=0, maxval=2, dtype=tf.dtypes.int32)
            - 1,
            x.dtype,
        )
        deterministic_part = -self._radius * tf.cast(tf.sign(x), x.dtype)
        return tf.where(tf.equal(x, 0), random_part, deterministic_part)

    def qmo(self, x, a):
        super().qmo(x, a)
        return tf.sign(x) * tf.math.minimum(tf.math.abs(x), self._radius / a)

    def shift_inside(self, x):
        super().shift_inside(x)
        x_norm = np.linalg.norm(x.flatten(), ord=self.p)
        if x_norm > self._radius:
            return self._radius * x / x_norm
        return x

    def euclidean_project(self, x):
        """Projects x to the closest (i.e. in L2-norm) point on the LpBall (p = 1, 2, inf) with radius r."""
        super().euclidean_project(x)
        x_norm = tf.norm(x, ord=np.inf)
        proj_x_fn = lambda: (
            tf.clip_by_value(x, -self._radius, self._radius),
            tf.constant(True, dtype=tf.bool),
        )
        x_fn = lambda: (x, tf.constant(False, dtype=tf.bool))
        return tf.cond(x_norm > self._radius, proj_x_fn, x_fn)


class Unconstrained(Constraint):
    """
    Parent/Base class for unconstrained parameter spaces
    :param n: dimension of unconstrained parameter space
    """

    def __init__(self, n):
        super().__init__(n)
        self._diameter = np.inf

    def lmo(self, x):
        super().__init__(x)
        raise NotImplementedError("no lmo for unconstrained parameters")

    def shift_inside(self, x):
        super().__init__(x)
        return x

    def euclidean_project(self, x):
        super().__init__(x)
        return x, tf.constant(False, dtype=tf.bool)


def make_feasible(model, constraints):
    """Shift all model parameters inside the feasible region defined by constraints"""
    trainable_vars_to_constraints = dict()

    for var, constraint in zip(model.trainable_variables, constraints):
        trainable_vars_to_constraints[var._shared_name] = constraint

    complete_constraints = []

    for var in model.variables:
        complete_constraints.append(
            trainable_vars_to_constraints.get(
                var._shared_name, Unconstrained(tf.size(var).numpy())
            )
        )

    counter = 0
    for layer in model.layers:
        new_weights = []
        for w in layer.get_weights():
            new_weights.append(complete_constraints[counter].shift_inside(w))
            counter += 1
        layer.set_weights(new_weights)
