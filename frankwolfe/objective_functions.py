import numpy as np

import pickle

import miniball
from scipy.sparse import issparse

import time

from abc import ABC, abstractmethod



# import cyminiball as miniball_fast
from frankwolfe.auxiliary_functions import rvs

class _ObjectiveFunction(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def grad(self, x):
        pass

    @abstractmethod
    def f(self, x):
        pass
    
    def line_search(self, x, d):
        raise NotImplementedError(
            "The feasible region does not have a method to perform a projection"
        )

    def largest_eigenvalue_hessian(
        self, 
    ):
        raise NotImplementedError(
            "The feasible region does not have a method to perform a projection"
        )
        
    def smallest_eigenvalue_hessian(
        self, 
    ):
        raise NotImplementedError(
            "The feasible region does not have a method to perform a projection"
        )

def load_pickled_object(filepath, parent_type=None):
    with open(filepath, "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object


def dump_pickled_object(filepath, target_object):
    with open(filepath, "wb") as f:
        pickle.dump(target_object, f)

# Example function: optimal experiment design over the simplex
class optimal_experiment_design:
    def __init__(self, A):
        self.p, self.n = A.shape
        self.A = A
        return

    # Evaluate function.
    def f(self, x):
        sign, value = np.linalg.slogdet(self.A.T.dot(np.multiply(x[:, None], self.A)))
        return -sign * value

    # Evaluate gradient.
    def grad(self, x):
        inv_matrix = np.linalg.inv(self.A.T.dot(np.multiply(x[:, None], self.A)))
        grad = np.zeros(self.p)
        for i in range(self.p):
            grad[i] = -self.A[i, :].T.dot(inv_matrix).dot(self.A[i, :])
        return grad

    # Note that this only applies to the simplex!
    # Update the quantities after taking the step.
    def line_search(self, x, d):
        inv_matrix = np.linalg.inv(self.A.T.dot(np.multiply(x[:, None], self.A)))
        vertex = (d + x).dot(self.A)
        mat_prod = vertex.T.dot(inv_matrix).dot(vertex)
        return (1 / self.n * mat_prod - 1) / (mat_prod - 1)

    # Check if point is in the domain of the function
    def in_domain(self, x):
        sign, value = np.linalg.slogdet(self.A.T.dot(np.multiply(x[:, None], self.A)))
        return sign != 0


class coreset_MEB:
    def __init__(self, A):
        self.A = A
        # self.A = self.A.astype(np.float64)
        self.d, self.n = self.A.shape
        self.b = np.zeros(self.n)
        for i in range(self.n):
            self.b[i] = np.linalg.norm(A[:, i]) ** 2
        return

    def f(self, x):
        return -(self.b.dot(x) - np.linalg.norm(self.A.dot(x)) ** 2)

    def grad(self, x):
        return -(self.b - 2 * self.A.T.dot(self.A.dot(x)))

    def grad_center(self, center):
        return -(self.b - 2 * self.A.T.dot(center))

    def line_search(self, x, d):
        return d.T.dot(self.grad(x)) / (-2 * np.linalg.norm(self.A.dot(x)) ** 2)

    def bounding_ball(self, x):
        indices = np.where(x > 0.0)[0]
        start = time.process_time()
        C, r2 = miniball.get_bounding_ball(self.A[:, indices].T)
        end = time.process_time()
        return C, r2, end - start, len(indices)

    def bounding_ball_indices(self, indices):
        start = time.process_time()
        C, r2 = miniball.get_bounding_ball(self.A[:, indices].T)
        end = time.process_time()
        return C, r2, end - start, len(indices)


# Example function: Quadratic in 2D.
class quadratic_2D(_ObjectiveFunction):
    import numpy as np

    def __init__(self, M):
        self.M = M.copy()
        return

    # Evaluate function.
    def f(self, x):
        return 0.5 * np.dot(x, np.dot(self.M, x))

    # Evaluate gradient.
    def grad(self, x):
        return np.dot(x, self.M)

    def returnM(self):
        return self.M

    def returnb(self):
        return np.zeros(2)

    # x denotes the initial point and d the direction.
    # If output is negative, d is probably pointing along gradient.
    def line_search(self, x, d):
        return -np.dot(self.grad(x), d) / np.dot(d, np.dot(self.M, d))



class quadratic(_ObjectiveFunction):
    import numpy as np

    def __init__(self, matrix, vector):
        self.len = matrix.shape[0]
        self.M = matrix.copy()
        self.b = vector.copy()
        eig, eigv = np.linalg.eig(self.M)
        self.L = np.real(max(eig))
        self.Mu = np.real(min(eig))
        return

    # Evaluate function.
    def f(self, x):
        return 0.5 * np.dot(x, np.dot(self.M, x)) + np.dot(self.b, x)

    # Evaluate gradient.
    def grad(self, x):
        return np.dot(x, self.M) + self.b

    # Return largest eigenvalue.
    def largest_eigenvalue_hessian(self):
        return self.L

    # Return smallest eigenvalue.
    def smallest_eigenvalue_hessian(self):
        return self.Mu

    def returnM(self):
        return self.M

    def returnb(self):
        return self.b

    # x denotes the initial point and d the direction.
    # If output is negative, d is probably pointing along gradient.
    def line_search(self, x, d):
        return -np.dot(self.grad(x), d) / np.dot(d, np.dot(self.M, d))


class quadratic_diagonal(_ObjectiveFunction):
    import numpy as np

    def __init__(self, size, xOpt, Mu=1.0, L=2.0):
        self.len = size
        self.matdim = int(np.sqrt(size))
        self.eigenval = np.zeros(size)
        self.eigenval[0] = Mu
        self.eigenval[-1] = L
        self.eigenval[1:-1] = np.random.uniform(Mu, L, size - 2)
        self.L = L
        self.Mu = Mu
        self.xOpt = xOpt
        self.b = -np.multiply(self.xOpt, self.eigenval)
        return

    def dim(self):
        return self.len

    # Evaluate function.
    def f(self, x):
        return 0.5 * np.dot(x, np.multiply(self.eigenval, x)) + np.dot(self.b, x)

    # Evaluate gradient.
    def grad(self, x):
        return np.multiply(x, self.eigenval) + self.b

    # Return largest eigenvalue.
    def largest_eigenvalue_hessian(self):
        return self.L

    # Return smallest eigenvalue.
    def smallest_eigenvalue_hessian(self):
        return self.Mu

    # Line Search.
    def line_search(self, x, d):
        return -np.dot(self.grad(x), d) / np.dot(d, np.multiply(self.eigenval, d))

        # Return largest eigenvalue.

    def returnM(self):
        return self.eigenval

    # Return smallest eigenvalue.
    def returnb(self):
        return self.b



class quadratic_sparse_signal_recovery(_ObjectiveFunction):
    import numpy as np

    def __init__(self, matrix, vector, alpha=0.0):
        self.len = matrix.shape[0]
        self.M = matrix.copy()
        self.b = vector.copy()
        self.alpha = alpha
        eig, eigv = np.linalg.eig(
            2 * self.M.T.dot(self.M) + 2.0 * self.alpha * np.eye(matrix.shape[1])
        )
        self.L = np.real(max(eig))
        self.Mu = max(np.real(min(eig)), 0.0)
        return

    # Evaluate function.
    def f(self, x):
        return (
            np.linalg.norm(self.b - self.M.dot(x)) ** 2
            + self.alpha * np.linalg.norm(x) ** 2
        )

    # Evaluate gradient.
    def grad(self, x):
        return -2 * self.M.T.dot(self.b - self.M.dot(x)) + 2 * self.alpha * x

    # Return largest eigenvalue.
    def largest_eigenvalue_hessian(self):
        return self.L

    # Return smallest eigenvalue.
    def smallest_eigenvalue_hessian(self):
        return self.Mu

    # x denotes the initial point and d the direction.
    # If output is negative, d is probably pointing along gradient.
    def line_search(self, x, d):
        if np.all(d == 0.0):
            return 0.0
        return -np.dot(self.grad(x), d) / (
            2 * np.linalg.norm(self.M.dot(d)) ** 2
            + 2 * self.alpha * np.linalg.norm(d) ** 2
        )


class quadratic_sparse_signal_recovery_probability_simplex(_ObjectiveFunction):
    import numpy as np

    def __init__(self, matrix, vector, alpha=0.0):
        self.len = matrix.shape[0]
        self.dim = matrix.shape[1]
        self.alpha = alpha
        self.M = matrix.copy()
        self.b = vector.copy()
        eig, eigv = np.linalg.eig(self.M.T.dot(self.M))
        aux_mat = 2 * self.M.T.dot(self.M) + 2.0 * self.alpha * np.eye(matrix.shape[1])
        eig, eigv = np.linalg.eig(np.block([[aux_mat, -aux_mat], [-aux_mat, aux_mat]]))
        self.L = np.real(max(eig))
        self.Mu = max(np.real(min(eig)) + self.alpha, 0.0)
        return

    # Evaluate function.
    def f(self, z):
        aux_vect1 = self.M.dot(z[: self.dim])
        aux_vect2 = self.M.dot(z[self.dim :])
        return (
            np.linalg.norm(aux_vect1) ** 2
            + np.linalg.norm(aux_vect2) ** 2
            - 2 * aux_vect1.dot(aux_vect2)
            + self.b.dot(2 * (-aux_vect1 + aux_vect2) + self.b)
            + self.alpha * np.linalg.norm(z[: self.dim] - z[self.dim :]) ** 2
        )

    # Evaluate gradient.
    def grad(self, z):
        aux_vect1 = self.M.dot(z[: self.dim] - z[self.dim :]) - self.b
        aux_vect2 = 2 * self.M.T.dot(aux_vect1) + 2 * self.alpha * (
            z[: self.dim] - z[self.dim :]
        )
        return np.concatenate((aux_vect2, -aux_vect2))

    # Return largest eigenvalue.
    def largest_eigenvalue_hessian(self):
        return self.L

    # Return smallest eigenvalue.
    def smallest_eigenvalue_hessian(self):
        return self.Mu

    # x denotes the initial point and d the direction.
    # If output is negative, d is probably pointing along gradient.
    def line_search(self, z, d):
        aux_vector = self.M.dot(d[self.dim :] - d[: self.dim])
        return (
            aux_vector.dot(self.M.dot(z[: self.dim] - z[self.dim :]) - self.b)
            - self.alpha
            * np.dot(d[: self.dim] - d[self.dim :], z[: self.dim] - z[self.dim :])
        ) / (
            aux_vector.dot(aux_vector)
            + self.alpha * np.linalg.norm(d[self.dim :] - d[: self.dim]) ** 2
        )


class logistic_regression(_ObjectiveFunction):
    import numpy as np

    def __init__(self, matrix, vector, L=1.0):
        self.len = vector.shape[0]
        self.M = matrix.copy()
        self.b = vector.copy()
        self.L = L
        return

    # Evaluate function.
    def f(self, x):
        # return 1/self.len*np.sum(np.log(1 + np.exp(-np.multiply(self.b, self.M.dot(x)))))
        return (
            1
            / self.len
            * np.sum(
                np.logaddexp(np.zeros(self.len), -np.multiply(self.b, self.M.dot(x)))
            )
        )

    # Evaluate gradient.
    def grad(self, x):
        return self.M.T.dot(
            -1
            / self.len
            * (self.b / (1.0 + np.exp(np.multiply(self.b, self.M.dot(x)))))
        )

    def full_grad_stochastic_naive(self, x):
        stoch_vector = np.zeros(len(x))
        for i in range(self.len):
            stoch_vector += -(
                self.b[i]
                * np.asarray(self.M[i].todense()).flatten()
                / (1.0 + np.exp(self.b[i] * self.M[i].dot(x)))
            )
        return stoch_vector / self.len

    def full_grad_stochastic(self, x):
        return self.grad(x)

    def stochastic_grad_naive(self, x, n):
        stoch_vector = np.zeros(len(x))
        for i in range(n):
            index = np.random.randint(0, self.len)
            stoch_vector += -(
                self.b[index]
                * np.asarray(self.M[index].todense()).flatten()
                / (1.0 + np.exp(self.b[index] * self.M[index].dot(x)))
            )
        return stoch_vector / n

    def stochastic_grad(self, x, n, max_mini_batch_size=int(1.0e6)):
        stoch_vector = np.zeros(len(x))
        for i in range(int(n // max_mini_batch_size)):
            indices = np.random.choice(self.len, size=max_mini_batch_size, replace=True)
            stoch_vector += -self.M[indices].T.dot(
                self.b[indices]
                / (1.0 + np.exp(np.multiply(self.b[indices], self.M[indices].dot(x))))
            )
        indices = np.random.choice(
            self.len, size=int(n % max_mini_batch_size), replace=True
        )
        stoch_vector += -self.M[indices].T.dot(
            self.b[indices]
            / (1.0 + np.exp(np.multiply(self.b[indices], self.M[indices].dot(x))))
        )
        return stoch_vector / n

    def stochastic_grad_CSFW_naive(self, x, n, old_alpha):
        new_alpha = old_alpha.copy()
        stoch_vector = np.zeros(len(x))
        for i in range(n):
            index = np.random.randint(0, self.len)
            new_alpha[index] = (
                1
                / self.len
                * (1.0 / (1.0 + np.exp(self.b[index] * self.M[index].dot(x))))
            )
            stoch_vector += (new_alpha[index] - old_alpha[index]) * (
                -self.b[index] * np.asarray(self.M[index].todense()).flatten()
            )
        return new_alpha, stoch_vector

    def stochastic_grad_CSFW(self, x, n, old_alpha, max_mini_batch_size=int(1.0e6)):
        new_alpha = old_alpha.copy()
        stoch_vector = np.zeros(len(x))
        for i in range(int(n // max_mini_batch_size)):
            indices = np.random.choice(self.len, size=max_mini_batch_size, replace=True)
            new_alpha[indices] = (
                1.0
                / (1.0 + np.exp(np.multiply(self.b[indices], self.M[indices].dot(x))))
            ) / self.len
            if issparse(self.M):
                stoch_vector += (
                    -self.M[indices]
                    .T.multiply(self.b[indices])
                    .dot(new_alpha[indices] - old_alpha[indices])
                )
            else:
                stoch_vector += -np.multiply(self.M[indices].T, self.b[indices]).dot(
                    new_alpha[indices] - old_alpha[indices]
                )
        indices = np.random.choice(
            self.len, size=int(n % max_mini_batch_size), replace=True
        )
        new_alpha[indices] = (
            1.0 / (1.0 + np.exp(np.multiply(self.b[indices], self.M[indices].dot(x))))
        ) / self.len
        if issparse(self.M):
            stoch_vector += (
                -self.M[indices]
                .T.multiply(self.b[indices])
                .dot(new_alpha[indices] - old_alpha[indices])
            )
        else:
            stoch_vector += -np.multiply(self.M[indices].T, self.b[indices]).dot(
                new_alpha[indices] - old_alpha[indices]
            )
        return new_alpha, stoch_vector

    def stochastic_grad_CSFW_backup(self, x, n, old_alpha):
        new_alpha = old_alpha.copy()
        indices = np.random.choice(self.len, size=n, replace=True)
        new_alpha[indices] = (
            1.0 / (1.0 + np.exp(np.multiply(self.b[indices], self.M[indices].dot(x))))
        ) / self.len
        stoch_vector = (
            -self.M[indices]
            .T.multiply(self.b[indices])
            .dot(new_alpha[indices] - old_alpha[indices])
        )
        return new_alpha, stoch_vector

    def stochastic_grad_SPIDER_naive(self, x, w, n):
        stoch_vector = np.zeros(len(x))
        for i in range(n):
            index = np.random.randint(0, self.len)
            stoch_vector += (
                -self.b[index]
                * np.asarray(self.M[index].todense()).flatten()
                * (
                    1.0 / (1.0 + np.exp(self.b[index] * self.M[index].dot(x)))
                    - 1.0 / (1.0 + np.exp(self.b[index] * self.M[index].dot(w)))
                )
            )
        return stoch_vector / n

    def stochastic_grad_SPIDER(self, x, w, n, max_mini_batch_size=int(1e6)):
        stoch_vector = np.zeros(len(x))
        for i in range(int(n // max_mini_batch_size)):
            indices = np.random.choice(self.len, size=max_mini_batch_size, replace=True)
            stoch_vector += -self.M[indices].T.dot(
                self.b[indices]
                * (
                    1.0
                    / (
                        1.0
                        + np.exp(np.multiply(self.b[indices], self.M[indices].dot(x)))
                    )
                    - 1.0
                    / (
                        1.0
                        + np.exp(np.multiply(self.b[indices], self.M[indices].dot(w)))
                    )
                )
            )
        indices = np.random.choice(
            self.len, size=int(n % max_mini_batch_size), replace=True
        )
        stoch_vector += -self.M[indices].T.dot(
            self.b[indices]
            * (
                1.0
                / (1.0 + np.exp(np.multiply(self.b[indices], self.M[indices].dot(x))))
                - 1.0
                / (1.0 + np.exp(np.multiply(self.b[indices], self.M[indices].dot(w))))
            )
        )
        return stoch_vector / n

    def stochastic_grad_1SFW(self, x, w):
        index = np.random.randint(0, self.len)
        if issparse(self.M):
            stoch_vector1 = (
                -self.b[index]
                * np.asarray(self.M[index].todense()).flatten()
                / (1.0 + np.exp(self.b[index] * self.M[index].dot(x)))
            )
            stoch_vector2 = (
                -self.b[index]
                * np.asarray(self.M[index].todense()).flatten()
                / (1.0 + np.exp(self.b[index] * self.M[index].dot(w)))
            )
        else:
            stoch_vector1 = (
                -self.b[index]
                * self.M[index]
                / (1.0 + np.exp(self.b[index] * self.M[index].dot(x)))
            )
            stoch_vector2 = (
                -self.b[index]
                * self.M[index]
                / (1.0 + np.exp(self.b[index] * self.M[index].dot(w)))
            )
        return stoch_vector1, stoch_vector2

    def largest_eigenvalue_hessian(self):
        return self.L

class quadratic_type_3(_ObjectiveFunction):
    import numpy as np

    def __init__(self, size, xOpt, Mu=1.0, L=2.0):

        self.len = size
        # Use a spectral decomposition to generate the matrix.
        eigenval = np.zeros(size)
        eigenval[0] = Mu
        eigenval[-1] = L
        eigenval[1:-1] = np.random.uniform(Mu, L, size - 2)
        self.M = np.zeros((size, size))
        A = rvs(size)
        for i in range(size):
            self.M += eigenval[i] * np.outer(A[i], A[i])
        self.L = L
        self.Mu = Mu
        # Define where the global optimum is.
        self.xOpt = xOpt
        self.b = -np.dot(self.xOpt, self.M)
        return

    # Evaluate function.
    def f(self, x):
        return 0.5 * np.dot(x, np.dot(self.M, x)) + np.dot(self.b, x)

    # Evaluate gradient.
    def grad(self, x):
        return np.dot(x, self.M) + self.b

    # x denotes the initial point and d the direction.
    # If output is negative, d is probably pointing along gradient.
    def line_search(self, x, d):
        return -np.dot(self.grad(x), d) / np.dot(d, np.dot(self.M, d))

    # Return largest eigenvalue.
    def largest_eigenvalue_hessian(self):
        return self.L

    # Return smallest eigenvalue.
    def smallest_eigenvalue_hessian(self):
        return self.Mu

        # Return largest eigenvalue.

    def returnM(self):
        return self.M

    def returnb(self):
        return self.b


class quadratic_stochastic(_ObjectiveFunction):
    import numpy as np

    def __init__(self, size, xOpt, Mu=1.0, L=2.0, sigma=1.0):
        

        self.len = size
        # Use a spectral decomposition to generate the matrix.
        eigenval = np.zeros(size)
        eigenval[0] = Mu
        eigenval[-1] = L
        eigenval[1:-1] = np.random.uniform(Mu, L, size - 2)
        self.M = np.zeros((size, size))
        A = rvs(size)
        for i in range(size):
            self.M += eigenval[i] * np.outer(A[i], A[i])
        self.L = L
        self.Mu = Mu
        # Define where the global optimum is.
        self.xOpt = xOpt
        self.b = -np.dot(self.xOpt, self.M)
        self.sigma = sigma
        return

    def f(self, x):
        return 0.5 * np.dot(x, np.dot(self.M, x)) + np.dot(self.b, x)

    def grad(self, x):
        return np.dot(x, self.M) + self.b

    def stochastic_grad(self, x, n, batch_size=int(1.0e6)):
        stoch_vector = np.zeros(len(x))
        for i in range(int(n // batch_size)):
            stoch_vector += np.sum(
                np.random.normal(
                    loc=0.0, scale=self.sigma, size=(self.len, batch_size)
                ),
                axis=1,
            )
        stoch_vector += np.sum(
            np.random.normal(
                loc=0.0, scale=self.sigma, size=(self.len, int(n % batch_size))
            ),
            axis=1,
        )
        return x * stoch_vector / n + stoch_vector / n + np.dot(x, self.M) + self.b

    def stochastic_grad_naive(self, x, n):
        aux = np.zeros(self.len)
        for _ in range(n):
            vector = np.random.normal(loc=0.0, scale=self.sigma, size=self.len)
            aux += x * vector + vector
        aux /= n
        return aux + np.dot(x, self.M) + self.b

    def stochastic_grad_SPIDER(self, x, w, n, batch_size=int(1.0e6)):
        stoch_vector = np.zeros(len(x))
        for _ in range(int(n // batch_size)):
            stoch_vector += np.sum(
                np.random.normal(
                    loc=0.0, scale=self.sigma, size=(self.len, batch_size)
                ),
                axis=1,
            )
        stoch_vector += np.sum(
            np.random.normal(
                loc=0.0, scale=self.sigma, size=(self.len, int(n % batch_size))
            ),
            axis=1,
        )
        return (x - w) * stoch_vector / n + np.dot(x - w, self.M)

    def stochastic_grad_SPIDER_naive(self, x, w, n):
        aux = np.zeros(self.len)
        for _ in range(n):
            vector = np.random.normal(loc=0.0, scale=self.sigma, size=self.len)
            aux += (x - w) * vector
        aux /= n
        return aux + np.dot(x - w, self.M)

    def stochastic_grad_1SFW(self, x, w):
        vector = np.random.normal(loc=0.0, scale=self.sigma, size=self.len)
        return (
            np.dot(x, self.M + np.diag(vector)) + self.b + vector,
            np.dot(w, self.M + np.diag(vector)) + self.b + vector,
        )

    def line_search(self, x, d):
        return -np.dot(self.grad(x), d) / np.dot(d, np.dot(self.M, d))

    def largest_eigenvalue_hessian(self):
        return self.L


# Minibatch Quadratic for the Stochastic algorithms.
# Function is sum over all i of 1/2*(x_i - w_i)^2, where w_i is given.
class quadratic_minibatch(_ObjectiveFunction):
    import numpy as np

    def __init__(self, dimension):
        self.w = np.random.rand(dimension)
        self.len = dimension
        return

    # Evaluate function.
    def f(self, x):
        return 0.5 * np.dot(x - self.w, x - self.w)

    # n is the number of samples we'll use for the gradient.
    # Gradients are iid, so the same sample can be used twice.
    def stochastic_grad(self, x, n):
        aux = np.zeros(self.len)
        for _ in range(n):
            index = np.random.randint(0, self.len)
            aux[index] += x[index] - self.w[index]
        aux /= n
        return aux

    # n is the number of samples we'll use for the gradient.
    # fullGradient is the full gradient at w
    def stochastic_variance_reduced_grad(self, x, w, n, fullGradient):
        aux = np.zeros(self.len)
        for _ in range(n):
            index = np.random.randint(0, self.len)
            aux[index] += x[index] - w[index]
        aux /= n
        return aux + fullGradient

    def grad(self, x):
        return x - self.w

    # Returns global optimum
    def optimum(self):
        return self.w

    # x denotes the initial point and d the direction.
    # If output is negative, d is probably pointing along gradient.
    def line_search(self, x, d):
        return -np.dot(self.gradFull(x), d) / np.dot(d, d)

    def largest_eigenvalue_hessian(self):
        return 1.0

    def Lipschitz(self):
        return 1.0


class projection_problem_function(_ObjectiveFunction):
    """
    Function class used in the CGS algorithm to compute an Euclidean projection
    of a point onto the feasible region using the FW algorithm.

    References
    ----------
    Lan, G., & Zhou, Y. (2016). Conditional gradient sliding for convex
    optimization. SIAM Journal on Optimization, 26(2), 1379-1409.

    """

    def __init__(self, y, gradient, beta):
        self.y = y
        self.gradient = gradient
        self.beta = beta

    def f(self, x):
        return (
            1
            / (2.0 * self.beta)
            * np.linalg.norm(self.gradient + self.beta * (x - self.y)) ** 2
        )

    def grad(self, x):
        return self.beta * (x - self.y) + self.gradient

    def line_search(self, x, d):
        return -np.dot(self.grad(x), d) / (self.beta * np.linalg.norm(d) ** 2)


# Function used in FCFW
class quadratic_function_over_simplex(_ObjectiveFunction):
    import numpy as np

    # Assemble the matrix from the active set.
    def __init__(self, activeSet, M, b):
        self.len = len(activeSet)
        # Mat = np.zeros((len(activeSet[0]), self.len))
        # self.b = np.zeros(len(activeSet))
        # for i in range(0, self.len):
        #     Mat[:, i] = activeSet[i]
        #     self.b[i] = np.dot(b, activeSet[i])

        Mat = np.vstack(activeSet).T
        self.b = Mat.T.dot(b)

        self.M = np.dot(Mat.T, np.dot(M, Mat))
        eig, vect = np.linalg.eig(self.M)
        self.L = max(eig)
        self.Mu = min(eig)
        return

    def f(self, x):
        return 0.5 * np.dot(x, np.dot(self.M, x)) + np.dot(self.b, x)

    def grad(self, x):
        return np.dot(x, self.M) + self.b

    def FW_gap(self, x):
        grad = self.grad(x)
        v = np.zeros(len(x))
        minVert = np.argmin(grad)
        v[minVert] = 1.0
        return np.dot(grad, x - v)

    def return_b(self):
        return self.b

    def return_M(self):
        return self.M

    def smallest_eigenvalue_hessian(self):
        return self.Mu

    def largest_eigenvalue_hessian(self):
        return self.L

    def line_search(self, x, d):
        Md = np.dot(self.M, d)
        return (-np.dot(x, Md) - np.dot(self.b, d)) / (np.dot(d, Md))
