# Introduction 
This repository contains several Frank-Wolfe (a.k.a. Conditional Gradient) algorithms implemented in Python. A sister repository implemented in Julia can be found [here](https://github.com/ZIB-IOL/FrankWolfe.jl). These algorithms form part of a family of first-order constrained optimization algorithms often dubbed "projection-free", as they do not require being able to project onto the feasible region of the optimization problem. This code was used to generate the experiments and images in the [Conditional Gradient Methods survey](https://conditional-gradients.org/), and can serve as an accompanying resource to the survey text. 

# Getting Started
In order to install the package, run in the parent directory 

```python
python -m pip install .
```

# Algorithms included
Among the algorithms implemented in this repository are the:

1. Frank-Wolfe algorithms: This includes the vanilla Frank-Wolfe algorithm originally proposed by Marguerite Frank and Philip Wolfe, as well as a [lazy variant](https://arxiv.org/abs/1610.05120), a [boosted variant](https://arxiv.org/abs/2003.06369), and a [nearest extreme point oracle variant](https://arxiv.org/abs/2102.02029).
2. Away/Pairwise Frank-Wolfe algorithms: This includes the [away-step variant](https://link.springer.com/article/10.1007/BF01589445), the [pairwise-step variant](https://arxiv.org/abs/1511.05932), and the [lazified away-step variant](https://arxiv.org/abs/1610.05120).
3. Decomposition-Invariant Pairwise Frank-Wolfe algorithms: This includes the [vanilla version](https://arxiv.org/abs/1605.06492) of this algorithm, and a [boosted variant](https://arxiv.org/abs/2003.06369) of the original algorithm. Note that both require the feasible region to be a 0-1 polytope.
4. Blended Conditional Gradients algorithms: This includes the [vanilla version](https://arxiv.org/abs/1805.07311) of this algorithm.
5. Conditional Gradient Sliding algorithms: This includes both the variant for convex functions, and for strongly convex functions originally proposed [here](https://optimization-online.org/2014/10/4605/).
6. Locally Accelerated Conditional Gradients algorithm: This algorithm was proposed [here](https://arxiv.org/abs/1906.07867), and in this repository we include an implementation of this algorithm for the probability simplex.

The aforementioned algorithms require being able to compute the gradient of the objective function, and being able to solve an LP over the feasible region (or being able to compute the nearest extreme point oracle). In the case where computing the gradient of the objective function is computationally expensive, and one has to resort to computing a stochastic approximation of the gradient, the following stochastic Frank-Wolfe algorithms can be used:

1. Stochastic Frank-Wolfe algorithm: Which includes the [first stochastic algorithm](https://arxiv.org/abs/1602.02101) proposed for the Frank-Wolfe family.
2. Stochastic Frank-Wolfe algorithms that use one stochastic gradient oracle call per iteration: This includes the [Momentum Stochastic Frank-Wolfe algorithm](https://arxiv.org/abs/1804.09554), as well as the [One-Sample Stochastic Frank-Wolfe algorithm](https://arxiv.org/abs/1910.04322)/.
3. Variance Reduced Stochastic Frank-Wolfe algorithms: Such as the [first variance reduced stochastic variant](https://arxiv.org/abs/1602.02101), or the [SPIDER Frank-Wolfe variant](https://proceedings.mlr.press/v97/yurtsever19b.html)
4. Other Stochastic Frank-Wolfe variants: Such as the [Stochastic Conditional Gradient Sliding algorithm](https://optimization-online.org/2014/10/4605/), or the [Constant Batch Size Stochastic Frank-Wolfe algorithm](https://arxiv.org/abs/2002.11860) for finite sum minimization.

Several application specific algorithms are included, like the vanilla Frank-Wolfe algorithm and the Away-step Frank-Wolfe algorithm for optimal experiment design, or the vanilla Frank-Wolfe algorithm, the Away-step Frank-Wolfe algorithm, and the Fully-Corrective Frank-Wolfe algorithm for the minimum enclosing ball problem.

# Available stepsizes
In the following we denote the stepsize at iteration $t$ by $\gamma_t$. The following stepsizes are available in this package:

1. Constant stepsizes of the form $\gamma_t = \alpha/(t + \beta)$, where $\alpha$ and $\beta$ are a user-defined multiplicative and additive constants, respectively. Typical choices for these are $\alpha = 2$ and $\beta = 2$, which was the original choice proposed by Marguerite Frank and Philip Wolfe.
2. Short-step of the form $\min \{ \left\langle \nabla f(x_t), x_t - v_t \right\rangle /(LD^2), 1 \}$, where $L$ is the smoothness parameter of the objective function $f$, $D$ is the diameter of the feasible region, and $v_t$ is the Frank-Wolfe vertex. See the [Conditional Gradient Methods survey](https://conditional-gradients.org/) for more details. Note that the parameters $L$ and $D$ have to be provided by the user.
3. Adaptive short-step stepsizes, that adapt to the local smoothness of the function, and which do not require providing any parameters.
4. Line search stepsizes that minimize $f$ between the line segment between $x_t$ and $v_t$. Note that this line search must be provided by the user for the specific objective function being minimized.

# Key Ingredients
When defining an objective function for use with the algorithms in the package, an _ObjectiveFunction Abstract Base Class has to be defined, with two necessary methods, one to call a zeroth-order oracle (by using f(x)), and a first order oracle, (by using grad(x)). For example, one could define the following objective function

```python
from frankwolfe.objective_functions import _ObjectiveFunction

class euclidean_norm(_ObjectiveFunction):
    def __init__(self):
        return

    # Evaluate function.
    def f(self, x):
        return 0.5*x.dot(x)

    # Evaluate gradient.
    def grad(self, x):
        return x
		
    # Return smoothness parameter of the function.
    def largest_eigenvalue_hessian(self):
        return 1.0

    # Return strong convexity paremeter of the function.
    def smallest_eigenvalue_hessian(self):
        return 1.0
		
```

Some speficic algorithms, or step size strategies, require knowledge of the smoothness parameter, or the strong convexity parameter of the objective function, this requires defining two additional methods in the _ObjectiveFunction, called largest_eigenvalue_hessian and smallest_eigenvalue_hessian respectively, which return the smoothness parameter of the function, and the strong convexity parameter of the function, shown in the example above. When defining a feasible region for use with the algorithms in the package, a _FeasibleRegion Abstract Base Class has to be defined, with one necessary methods, in order to call the linear minimization oracle (using linear_optimization_oracle(x)). For example, one could define the following objective function:

```python
from frankwolfe.objective_functions import _FeasibleRegion

class L1_ball(_FeasibleRegion):
    def __init__(self, dimension, alpha=1.0):
        self.len = dimension
        self.alpha = alpha
        return

    # Solve an LP over the L1 ball
    def linear_optimization_oracle(self, grad):
        v = np.zeros(len(grad), dtype=float)
        maxInd = np.argmax(np.abs(grad))
        v[maxInd] = -self.alpha * np.sign(grad[maxInd])
        return v
		
    # Return the diameter of the L1 ball
    def diameter(self):
        return 2.0 * self.alpha


```

In certain cases, the algorithms included in the package require knowledge of the diameter of the feasible region, in which case one has to also define a diameter method in the _ObjectiveFunction Abstract Base Class.

