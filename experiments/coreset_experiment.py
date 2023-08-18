# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:50:30 2022

@author: pccom
"""
import numpy as np
import miniball
import time
import sys
import os
from os.path import join
import open3d as o3d
import pickle

from frankwolfe.application_specific_algorithms import (
    frank_wolfe_minimum_enclosing_ball,
    away_frank_wolfe_minimum_enclosing_ball,
    fully_corrective_frank_wolfe_minimum_enclosing_ball,
)
from frankwolfe.feasible_regions import probability_simplex
from frankwolfe.objective_functions import coreset_MEB


# Path to the bunny ply file
bunny_ply_file = join("Data", "bun_zipper.ply")

# Read the point cloud and draw the bunny
def get_bunny_mesh():
    mesh = o3d.io.read_triangle_mesh(bunny_ply_file)
    mesh.compute_vertex_normals()
    return mesh


mesh_bunny = get_bunny_mesh()
S = np.asarray(mesh_bunny.vertices)


# # # Compute the minimum bounding ball with all the data
start_total = time.process_time()
C_total, radius2_total = miniball.get_bounding_ball(S)
end_total = time.process_time()
total_time = end_total - start_total
print(total_time, radius2_total)

function = coreset_MEB(S.T)
dimension, _ = S.shape
LPOracle = probability_simplex(dimension)
initial_point = np.zeros(dimension)
initial_point[0] = 1
num_steps = 10000
# num_steps = 10

data_FCFW_coreset = fully_corrective_frank_wolfe_minimum_enclosing_ball(
    initial_point,
    function,
    LPOracle,
    radius2_total,
    algorithm_parameters={
        "maximum_time": 7200.0,
        "maximum_iterations": num_steps,
    },
)

data_AFW_coreset = away_frank_wolfe_minimum_enclosing_ball(
    initial_point,
    function,
    LPOracle,
    algorithm_parameters={
        "maximum_time": 7200.0,
        "maximum_iterations": num_steps,
    },
)

data_FW_coreset = frank_wolfe_minimum_enclosing_ball(
    initial_point,
    function,
    LPOracle,
    algorithm_parameters={
        "maximum_time": 7200.0,
        "maximum_iterations": num_steps,
    },
)

# Put the results in a picle object and then output
data_FW_coreset = {
    "name": r"FW_coreset_MEB",
    "f_value": data_FW_coreset["function_eval"],
    "dual_gap": data_FW_coreset["frank_wolfe_gap"],
    "cardinality": data_FW_coreset["cardinality"],
    "radius2": data_FW_coreset["radius2"],
    "solution": data_FW_coreset["solution"],
    "time_radius": data_FW_coreset["time_radius"],
    "time": data_FW_coreset["timing"],
}
data_AFW_coreset = {
    "name": r"AFW_coreset_MEB",
    "f_value": data_AFW_coreset["function_eval"],
    "dual_gap": data_AFW_coreset["frank_wolfe_gap"],
    "cardinality": data_AFW_coreset["cardinality"],
    "radius2": data_AFW_coreset["radius2"],
    "solution": data_AFW_coreset["solution"],
    "time_radius": data_AFW_coreset["time_radius"],
    "time": data_AFW_coreset["timing"],
}
data_FCFW_coreset = {
    "name": r"FCFW_coreset_MEB",
    "f_value": data_FCFW_coreset["function_eval"],
    "dual_gap": data_FCFW_coreset["frank_wolfe_gap"],
    "cardinality": data_FCFW_coreset["cardinality"],
    "radius2": data_FCFW_coreset["radius2"],
    "time": data_FCFW_coreset["timing"],
}
data_exact = {
    "name": r"exact",
    "total_time": total_time,
    "optimal_radius": radius2_total,
}

results = {
    "exact": data_exact,
    "FW": data_FW_coreset,
    "AFW": data_AFW_coreset,
    "FCFW": data_FCFW_coreset,
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
filepath = os.path.join(os.getcwd(), "output_data", "coreset_MEB.pickle")
with open(filepath, "wb") as f:
    pickle.dump(results, f)

y_lim_cutoff = 1.0e-7
max_point_FCFW = 5


mesh_bunny = get_bunny_mesh()
S = np.asarray(mesh_bunny.vertices)

# Read the point cloud and draw the bunny
mesh_bunny.compute_vertex_normals()

# Draw the point cloud and the key points
cloud = o3d.io.read_point_cloud(bunny_ply_file)  # Read the point cloud
indices = np.where(results["FW"]["solution"] > 0.0)[0]
key_points = S[indices, :]

# # Try to draw bigger points.
# radius_key_points = 0.005
# list_spheres = [mesh_bunny]
# for i in range(len(indices)):
#     mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius = radius_key_points).translate(key_points[i,:])
#     mesh_sphere.compute_vertex_normals()
#     mesh_sphere.paint_uniform_color([0.0, 0.0, 0.0])
#     list_spheres.append(mesh_sphere)
# # o3d.visualization.draw_geometries(list_spheres)
# o3d.visualization.draw_geometries([mesh_bunny])

# # Draw the minimum enclosing ball
# center = S.T.dot(data['FW']['solution'])
# radius = np.sqrt(-data['FW']['function_eval'][-1])
# mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius = radius).translate(center)
# mesh_sphere.compute_vertex_normals()
# mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
# o3d.visualization.draw_geometries([mesh_sphere], mesh_show_wireframe  = True, mesh_show_back_face = False)

figure_size = 7.3

output_filepath = os.path.join(
    output_directory_images,
    "coreset_MEB_primal_iteration.pdf",
)

list_x_label = [
    np.arange(len(results["FW"]["f_value"])) + 1,
    np.arange(len(results["AFW"]["f_value"])) + 1,
    np.arange(len(results["FCFW"]["f_value"])) + 1,
]
list_data = [
    np.asarray([x + results["exact"]["optimal_radius"] for x in results["FW"]["f_value"]]),
    np.asarray([x + results["exact"]["optimal_radius"] for x in results["AFW"]["f_value"]]),
    np.asarray([-x + results["exact"]["optimal_radius"] for x in results["FCFW"]["radius2"]]),
]
list_legend = [r"$\mathrm{FW}$", r"$\mathrm{AFW}$", r"$\mathrm{FCFW}$"]

from frankwolfe.plotting_function import plot_results

colors = ["k", "c", "m"]
markers = ["o", "s", "P"]
colors_fill = ["None", "None", "None"]


plot_results(
    list_x_label,
    list_data,
    list_legend,
    "",
    r"$\mathrm{Iteration \ (t)}$",
    r"$f(x^*) - f(x_t)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    y_limits=[
        y_lim_cutoff,
        None,
    ],
    x_limits=[
        1.0,
        1.0
        + max(
            len(results["FW"]["f_value"]),
            len(results["AFW"]["f_value"]),
            len(results["FCFW"]["f_value"]),
        ),
    ],
    # legend_location="lower left",
    save_figure=output_filepath,
    # save_figure=None,
    figure_width=figure_size,
)


output_filepath = os.path.join(
    output_directory_images,
    "coreset_MEB_radius_iteration_small.pdf",
)

list_x_label = [
    np.arange(len(results["FW"]["radius2"])) + 1,
    np.arange(len(results["AFW"]["radius2"])) + 1,
    np.arange(len(results["FCFW"]["radius2"])) + 1,
]
list_data = [
    np.asarray([-x + results["exact"]["optimal_radius"] for x in results["FW"]["radius2"]]),
    np.asarray([-x + results["exact"]["optimal_radius"] for x in results["AFW"]["radius2"]]),
    np.asarray([-x + results["exact"]["optimal_radius"] for x in results["FCFW"]["radius2"]]),
]
# list_legend = ["FW", "AFW", "FCFW"]
list_legend = [r"$\mathrm{FW}$", r"$\mathrm{AFW}$", r"$\mathrm{FCFW}$"]

colors = ["k", "c", "m"]
markers = ["o", "s", "P"]
colors_fill = ["None", "None", "None"]


plot_results(
    list_x_label,
    list_data,
    [],
    "",
    r"$\mathrm{Iteration \ (t)}$",
    r"$f(x^*) - f(x^{\mathcal{N}_t})$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=True,
    y_limits=[
        1.0e-13,
        None,
    ],
    x_limits=[
        1.0,
        1.0
        + max(
            len(results["FW"]["f_value"]),
            len(results["AFW"]["f_value"]),
            # len(data["FCFW"]["f_value"]),
        ),
    ],
    # legend_location="lower left",
    save_figure=output_filepath,
    # save_figure=None,
    figure_width=figure_size,
)


output_filepath = os.path.join(
    output_directory_images,
    "coreset_MEB_cardinality.pdf",
)

markers = ["None", "None", "None"]
colors_fill = ["None", "None", "None"]

list_data = [
    results["FW"]["cardinality"],
    results["AFW"]["cardinality"],
    results["FCFW"]["cardinality"],
]

plot_results(
    list_x_label,
    list_data,
    [],
    "",
    r"$\mathrm{Iteration \ (t)}$",
    r"$\mid\mathcal{N}_t\mid$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=True,
    log_y=False,
    # legend_location="lower left",
    save_figure=output_filepath,
    # save_figure=None,
    figure_width=figure_size,
)

output_filepath = os.path.join(
    output_directory_images,
    "coreset_MEB_primal_time.pdf",
)

markers = ["o", "s", "P"]

list_x_label = [
    results["FW"]["time"],
    results["AFW"]["time"],
    results["FCFW"]["time"],
]
list_data = [
    np.asarray([x + results["exact"]["optimal_radius"] for x in results["FW"]["f_value"]]),
    np.asarray([x + results["exact"]["optimal_radius"] for x in results["AFW"]["f_value"]]),
    np.asarray([-x + results["exact"]["optimal_radius"] for x in results["FCFW"]["f_value"]]),
]
plot_results(
    list_x_label,
    list_data,
    [],
    "",
    r"$\mathrm{Time \ (s)}$",
    r"$f(x^*) - f(x_t)$",
    colors,
    markers,
    colorfills=colors_fill,
    log_x=False,
    log_y=True,
    y_limits=[
        y_lim_cutoff,
        None,
    ],
    x_limits=[
        None,
        220,
    ],
    # legend_location="lower left",
    # save_figure=output_filepath,
    save_figure=None,
    figure_width=figure_size,
)
