# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:09:44 2020

@author: pccom
"""
import os, sys
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from matplotlib import rc
import numpy as np
import matplotlib

rc("text", usetex=True)
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": [r"\usepackage{libertinust1math, libertinus}"],
    }
)
sys.path.append("..")

image_target_directory = os.path.join(os.getcwd(), "..", "..", "tex", "Images")

# -----------------------Non-linearly_separable .------------------------------
# Generate a series of 2D points from a Gaussian
mu_1 = np.asarray([0.5, 0.5])
mu_2 = np.asarray([-0.5, -0.5])
std_dev = 0.5 * np.identity(2)
number_samples = 30
regularization = 10000
random_points_1 = np.random.multivariate_normal(mu_1, std_dev, size=number_samples)
random_points_2 = np.random.multivariate_normal(mu_2, std_dev, size=number_samples)

# Solve SVM problem
from sklearn import svm

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=regularization)
clf.fit(
    np.vstack((random_points_1, random_points_2)),
    np.hstack((np.ones(number_samples), np.zeros(number_samples))),
)

plt.scatter(random_points_1[:, 0], random_points_1[:, 1])
plt.scatter(random_points_2[:, 0], random_points_2[:, 1], marker="s")

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)

plt.gca().set_aspect("equal", adjustable="box")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    os.path.join(image_target_directory, "SVM_soft_margin.pdf"),
    format="pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()
plt.close()

# -----------------------Non-linearly_separable .------------------------------
# Generate a series of 2D points from a Gaussian
mu_1 = np.asarray([0.5, 0.5])
mu_2 = np.asarray([-0.5, -0.5])
std_dev = 0.1 * np.identity(2)
number_samples = 30
regularization = 1000
random_points_1 = np.random.multivariate_normal(mu_1, std_dev, size=number_samples)
random_points_2 = np.random.multivariate_normal(mu_2, std_dev, size=number_samples)

# Solve SVM problem
from sklearn import svm

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=regularization)
clf.fit(
    np.vstack((random_points_1, random_points_2)),
    np.hstack((np.ones(number_samples), np.zeros(number_samples))),
)

plt.scatter(random_points_1[:, 0], random_points_1[:, 1])
plt.scatter(random_points_2[:, 0], random_points_2[:, 1], marker="s")

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)

plt.gca().set_aspect("equal", adjustable="box")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    os.path.join(image_target_directory, "SVM_hard_margin.pdf"),
    format="pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()
plt.close()
