# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:55:36 2019

@author: pccom
"""
import numpy as np
import sys, os
import pickle


from frankwolfe.algorithms import frank_wolfe, away_frank_wolfe, fully_corrective_frank_wolfe
from frankwolfe.objective_functions import quadratic_2D
from frankwolfe.feasible_regions import polytope_defined_by_vertices
from frankwolfe.auxiliary_functions import step_size_class, stopping_criterion

M = np.array([[2.0, 0.0], [0.0, 1.0]])
function = quadratic_2D(M)

vertices = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
feasible_region = polytope_defined_by_vertices(vertices)

initial_point = np.array([0.0, 1.0])
lambda_values = [1.0]
active_set = [initial_point]

# Run the four algorithms and store the data together.
data_FW_vanilla = frank_wolfe(
    function,
    feasible_region,
    algorithm_parameters={
        "algorithm_type": "standard",
        "return_points": True,
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": 14}),
    step_size =step_size_class("line_search"),
)

data_FW_away = away_frank_wolfe(
    function,
    feasible_region,
    initial_convex_decomposition=lambda_values,
    initial_active_set=active_set,
    algorithm_parameters={
        "algorithm_type": "standard",
        "return_points": True,
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": 6}),
    step_size =step_size_class("line_search"),
)

data_FW_pairwise = away_frank_wolfe(
    function,
    feasible_region,
    initial_convex_decomposition=lambda_values,
    initial_active_set=active_set,
    algorithm_parameters={
        "algorithm_type": "pairwise",
        "return_points": True,
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": 5}),
    step_size =step_size_class("line_search"),
)

data_FW_fully_corrective = fully_corrective_frank_wolfe(
    function,
    feasible_region,
    initial_convex_decomposition=lambda_values,
    initial_active_set=active_set,
    algorithm_parameters={
        "return_points": True,
    },
    x0 = initial_point,
    stopping_criteria=stopping_criterion({"iterations": 3}),
)

experiment_details = {
    "dimension": 2,
    "Mu": 1.0,
    "L": 2.0,
    "feasible_region": "triangle",
    "vertices": vertices,
}
results = {
    "details": experiment_details,
    "vanilla": data_FW_vanilla,
    "away": data_FW_away,
    "pairwise": data_FW_pairwise,
    "fully_corrective": data_FW_fully_corrective,
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
filepath = os.path.join(os.getcwd(), "output_data", "iterates_triangle_comparison.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

# Output also the function used in the experiment, as it will be used for the contour plot.
filepath = os.path.join(os.getcwd(), "output_data", "lower_bound_comparison.pickle")
with open(filepath, "wb") as f:
    pickle.dump(function, f)
    
    
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import matplotlib

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newpxtext, newpxmath, pifont, dsfont}",
    }
)

# -----------------------Vanilla FW Example.------------------------------

vertices = results["details"]["vertices"]
points = [np.asarray([0.0, 1.0])] + results["vanilla"]["x_val"][1:]
num_it = len(points)

fig = plt.figure()
ax = fig.add_subplot(111)

# Draw the feasible region.
t1 = plt.Polygon(vertices[:3, :], fill=False, linewidth=1.5, zorder=10)
ax.add_patch(t1)

# Crop the contourplot to the triangle.
# for artist in ax.get_children():
#    artist.set_clip_path(t1)

# Draw the path of the iterates.
plt.scatter(points[0][0], points[0][1], c="k", zorder=0)
for i in range(1, num_it):
    plt.scatter(points[i][0], points[i][1], c="k", zorder=0)
    verts = [
        (points[i - 1][0], points[i - 1][1]),  # left, bottom
        (points[i][0], points[i][1]),
    ]
    codes = [Path.MOVETO, Path.LINETO]
    path = Path(verts, codes)
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        linewidth=2.0,
        color="b",
        mutation_scale=10,
        zorder=0,
    )
    ax.add_artist(arrow)

# Draw the contour plot.
delta = 0.025
x = np.arange(-1.15, 1.15, delta)
y = np.arange(-0.15, 1.15, delta)
X, Y = np.meshgrid(x, y)
Z = np.zeros((len(y), len(x)))
for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = function.f(np.array([x[i], y[j]]))

CS = ax.contour(
    X,
    Y,
    Z,
    levels=np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]),
    linewidths=0.25,
    zorder=0,
)
plt.clabel(CS, inline=1, fmt="%1.1f", fontsize=10)

plt.text(0.0, 1.025, r"$x_0$", {"color": "k", "fontsize": 15}, zorder=0)
plt.text(
    points[1][0] - 0.12,
    points[1][1],
    r"$x_1$",
    {"color": "k", "fontsize": 15},
    zorder=30,
)
plt.text(
    points[2][0] + 0.03,
    points[2][1],
    r"$x_2$",
    {"color": "k", "fontsize": 15},
    zorder=30,
)
plt.text(
    points[3][0] - 0.12,
    points[3][1],
    r"$x_3$",
    {"color": "k", "fontsize": 15},
    zorder=30,
)
plt.text(
    points[4][0] + 0.03,
    points[4][1],
    r"$x_4$",
    {"color": "k", "fontsize": 15},
    zorder=30,
)
plt.text(
    points[5][0] - 0.12,
    points[5][1],
    r"$x_5$",
    {"color": "k", "fontsize": 15},
    zorder=30,
)
plt.text(
    points[6][0] + 0.03,
    points[6][1],
    r"$x_6$",
    {"color": "k", "fontsize": 15},
    zorder=30,
)
plt.text(
    points[7][0] - 0.12,
    points[7][1] - 0.05,
    r"$x_7$",
    {"color": "k", "fontsize": 15},
    zorder=30,
)
plt.text(0, -0.075, r"$x^*$", {"color": "r", "fontsize": 15}, zorder=20)

# Indicate the directions where the algorithm is going to.
plt.plot(
    [points[2][0], vertices[1][0]],
    [points[2][1], vertices[1][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[3][0], vertices[0][0]],
    [points[3][1], vertices[0][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[4][0], vertices[1][0]],
    [points[4][1], vertices[1][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[5][0], vertices[0][0]],
    [points[5][1], vertices[0][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[6][0], vertices[1][0]],
    [points[6][1], vertices[1][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[7][0], vertices[0][0]],
    [points[7][1], vertices[0][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[8][0], vertices[1][0]],
    [points[8][1], vertices[1][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[9][0], vertices[0][0]],
    [points[9][1], vertices[0][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[10][0], vertices[1][0]],
    [points[10][1], vertices[1][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[11][0], vertices[0][0]],
    [points[11][1], vertices[0][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[12][0], vertices[1][0]],
    [points[12][1], vertices[1][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)
plt.plot(
    [points[13][0], vertices[0][0]],
    [points[13][1], vertices[0][1]],
    "--b",
    linewidth=0.5,
    zorder=20,
)

# Draw the gradient.
vector = -function.grad(points[13])
verts = [
    (points[13][0], points[13][1]),
    (points[13][0] + 0.75 * vector[0], points[13][1] + 0.75 * vector[1]),
]
codes = [Path.MOVETO, Path.LINETO]
path = Path(verts, codes)
arrow = FancyArrowPatch(
    path=path, arrowstyle="-|>", linewidth=2.0, color="c", mutation_scale=10, zorder=0
)
ax.add_artist(arrow)
plt.text(
    points[13][0] - 0.25,
    points[13][1] - 0.1,
    r"$-\nabla f(x_{14})$",
    {"color": "c", "fontsize": 12},
    zorder=30,
)

# Indicate the minimizer.
plt.scatter(0, 0, c="r")
# ax.annotate(r'$x^*$', (0, -0.05))
plt.gca().set_aspect("equal", adjustable="box")
plt.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig(
    os.path.join(output_directory_images, "Vanilla.pdf"),
    format="pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close()


# # -----------------------Away FW Example.------------------------------

points = [np.asarray([0.0, 1.0])] + results["away"]["x_val"][1:]
num_it = len(points)

fig = plt.figure()
ax = fig.add_subplot(111)

# Draw the feasible region.
t1 = plt.Polygon(vertices[:3, :], fill=False, linewidth=1.5, zorder=10)
ax.add_patch(t1)

# Indicate the away step.
plt.plot(
    [points[6][0], points[0][0]],
    [points[6][1], points[0][1]],
    ":r",
    linewidth=1.0,
    zorder=0,
)

# Draw the path of the iterates.
plt.scatter(points[0][0], points[0][1], c="k", zorder=0)
for i in range(1, num_it):
    plt.scatter(points[i][0], points[i][1], c="k", zorder=0)
    verts = [
        (points[i - 1][0], points[i - 1][1]),  # left, bottom
        (points[i][0], points[i][1]),
    ]
    codes = [Path.MOVETO, Path.LINETO]
    path = Path(verts, codes)
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        linewidth=2.0,
        color="b",
        mutation_scale=10,
        zorder=0,
    )
    ax.add_artist(arrow)

# Draw the contour plot.
delta = 0.025
x = np.arange(-1.15, 1.15, delta)
y = np.arange(-0.15, 1.15, delta)
X, Y = np.meshgrid(x, y)
Z = np.zeros((len(y), len(x)))
for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = function.f(np.array([x[i], y[j]]))

CS = ax.contour(
    X,
    Y,
    Z,
    levels=np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]),
    linewidths=0.25,
    zorder=0,
)
plt.clabel(CS, inline=1, fmt="%1.1f", fontsize=10)


plt.text(0.0, 1.025, r"$x_0$", {"color": "k", "fontsize": 15}, zorder=0)
plt.text(
    points[1][0] - 0.12,
    points[1][1],
    r"$x_1$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[2][0] + 0.03,
    points[2][1],
    r"$x_2$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[3][0] - 0.12,
    points[3][1] - 0.05,
    r"$x_3$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[4][0] + 0.03,
    points[4][1],
    r"$x_4$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[5][0] - 0.12,
    points[5][1] - 0.05,
    r"$x_5$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[6][0] - 0.12,
    points[6][1] - 0.075,
    r"$x_6$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(0, -0.075, r"$x^*$", {"color": "r", "fontsize": 15}, zorder=20)

# Indicate the minimizer.
plt.scatter(0, 0, c="r")
plt.gca().set_aspect("equal", adjustable="box")
plt.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig(
    os.path.join(output_directory_images, "Away.pdf"),
    format="pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close()

# -----------------------Pairwise FW Example.------------------------------

points = [np.asarray([0.0, 1.0])] + results["pairwise"]["x_val"][1:]
num_it = len(points)

fig = plt.figure()
ax = fig.add_subplot(111)

# Draw the feasible region.
t1 = plt.Polygon(vertices[:3, :], fill=False, linewidth=1.0, zorder=10)
ax.add_patch(t1)

# Crop the contourplot to the triangle.
# for artist in ax.get_children():
#    artist.set_clip_path(t1)

# Draw the path of the iterates.
plt.scatter(points[0][0], points[0][1], c="k", zorder=0)
for i in range(1, num_it):
    plt.scatter(points[i][0], points[i][1], c="k", zorder=0)
    verts = [
        (points[i - 1][0], points[i - 1][1]),  # left, bottom
        (points[i][0], points[i][1]),
    ]
    codes = [Path.MOVETO, Path.LINETO]
    path = Path(verts, codes)
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        # linestyle="--",
        linewidth=2.0,
        color="b",
        mutation_scale=10,
        zorder=0,
    )
    ax.add_artist(arrow)

# Draw the contour plot.
delta = 0.025
x = np.arange(-1.15, 1.15, delta)
y = np.arange(-0.15, 1.15, delta)
X, Y = np.meshgrid(x, y)
Z = np.zeros((len(y), len(x)))
for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = function.f(np.array([x[i], y[j]]))

CS = ax.contour(
    X,
    Y,
    Z,
    levels=np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]),
    linewidths=0.25,
    zorder=0,
)
plt.clabel(CS, inline=1, fmt="%1.1f", fontsize=10)

plt.text(0.0, 1.025, r"$x_0$", {"color": "k", "fontsize": 15}, zorder=0)
plt.text(
    points[1][0] - 0.1,
    points[1][1],
    r"$x_1$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[2][0] + 0.03,
    points[2][1],
    r"$x_2$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[3][0] - 0.1,
    points[3][1],
    r"$x_3$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[4][0] - 0.1,
    points[4][1],
    r"$x_4$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[5][0] + 0.03,
    points[5][1],
    r"$x_5$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(0, -0.075, r"$x^*$", {"color": "r", "fontsize": 15}, zorder=20)

# Indicate the minimizer.
plt.scatter(0, 0, c="r")
# ax.annotate(r'$x^*$', (0, -0.05))
plt.gca().set_aspect("equal", adjustable="box")
plt.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig(
    os.path.join(output_directory_images, "Pairwise.pdf"),
    format="pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close()

# -----------------------Fully Corrective FW Example.------------------------------

points = [np.asarray([0.0, 1.0])] + results["fully_corrective"]["x_val"][1:]
num_it = len(points)

fig = plt.figure()
ax = fig.add_subplot(111)

# Draw the feasible region.
t1 = plt.Polygon(vertices[:3, :], fill=False, linewidth=1.0, zorder=10)
ax.add_patch(t1)

# Draw the path of the iterates.
plt.scatter(points[0][0], points[0][1], c="k", zorder=0)
for i in range(1, 3):
    plt.scatter(points[i][0], points[i][1], c="k", zorder=0)
    verts = [
        (points[i - 1][0], points[i - 1][1]),  # left, bottom
        (points[i][0], points[i][1]),
    ]
    codes = [Path.MOVETO, Path.LINETO]
    path = Path(verts, codes)
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        # linestyle="--",
        linewidth=2.0,
        color="b",
        mutation_scale=15,
        zorder=0,
    )
    ax.add_artist(arrow)

# Draw the contour plot.
delta = 0.025
x = np.arange(-1.15, 1.15, delta)
y = np.arange(-0.15, 1.15, delta)
X, Y = np.meshgrid(x, y)
Z = np.zeros((len(y), len(x)))
for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = function.f(np.array([x[i], y[j]]))

CS = ax.contour(
    X,
    Y,
    Z,
    levels=np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]),
    linewidths=0.25,
    zorder=0,
)

plt.clabel(CS, inline=1, fmt="%1.1f", fontsize=10)

plt.text(0.0, 1.025, r"$x_0$", {"color": "k", "fontsize": 15}, zorder=0)
plt.text(
    points[1][0] - 0.1,
    points[1][1],
    r"$x_1$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(
    points[2][0] + 0.03,
    points[2][1] + 0.025,
    r"$x_2$",
    {"color": "k", "fontsize": 15},
    zorder=20,
)
plt.text(0, -0.075, r"$x^*$", {"color": "r", "fontsize": 15}, zorder=20)

# Indicate the minimizer.
plt.scatter(0, 0, c="r")
# ax.annotate(r'$x^*$', (0, -0.05))
plt.gca().set_aspect("equal", adjustable="box")
plt.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig(
    os.path.join(output_directory_images, "FCFW.pdf"),
    format="pdf",
    bbox_inches="tight",
    transparent=True,
)