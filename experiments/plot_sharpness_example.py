# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:00:00 2021

@author: pccom
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import rc
import matplotlib

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newpxtext, newpxmath, pifont, dsfont}",
    }
)


plt.rcParams.update({"font.size": 19})
linewidth_figures = 4.0
fontsize = 20
fontsize_legend = 20
size_marker = 12

output_directory_images = os.path.join(os.getcwd(), "output_images")
# Make the directory if it does not already exist.
if not os.path.isdir(output_directory_images):
    os.makedirs(output_directory_images)

output_filepath = os.path.join(
    output_directory_images,
    "sharpness_example.pdf",
)

figure_size = 8

min_x = -1.0
max_x = 1.0
number_samples = 1000
c = 1.0
theta_parameters = [0.25, 0.5, 0.75, 1.0]
label_list = [
    r"$\theta = 0.25$",
    r"$\theta = 0.5$",
    r"$\theta = 0.75$",
    r"$\theta = 1.0$",
]
marker_scheme = ["o", "s", "^", "P"]
color_scheme = ["k", "b", "g", "r"]
colors_fill = ["None", "None", "None", "None"]

x_opt = 0.0
f_val_opt = 0.0

x = np.linspace(min_x, max_x, number_samples + 1)
list_y_values = []
list_x_values = []
for j in range(len(theta_parameters)):
    fun_val = np.zeros(number_samples + 1)
    for i in range(number_samples + 1):
        fun_val[i] = np.power(
            1 / c * np.linalg.norm(x[i] - x_opt), 1 / theta_parameters[j]
        )
    list_x_values.append(x)
    list_y_values.append(fun_val)

from frankwolfe.plotting_function import plot_results

plot_results(
    list_x_values,
    list_y_values,
    label_list,
    "",
    r"$x$",
    r"$f(x) - f(x^*)$",
    color_scheme,
    marker_scheme,
    colorfills=colors_fill,
    log_x=False,
    log_y=False,
    save_figure=output_filepath,
    figure_width=figure_size,
)
