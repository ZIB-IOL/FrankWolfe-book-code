# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:16:41 2020

@author: pccom
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed

import matplotlib


fontsize = 10
linewidth_figures = 1.0
linewidth_markers = 1.0
framealpha_val = 0.4
label_spacing_val = 0.10
borderpad_val = 0.05


matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newpxtext, newpxmath, pifont, dsfont}",
    }
)


cm = 1 / 2.54  # centimeters in inches
aspect_ratio = 6.4 / 4.8


def plot_results(
    list_x,
    list_data,
    list_legend,
    title,
    x_label,
    y_label,
    colors,
    markers,
    colorfills=None,
    log_x=True,
    log_y=True,
    save_figure=None,
    legend_location=None,
    outside_legend=False,
    x_limits=None,
    y_limits=None,
    label_font_size=8,
    number_columns_legend=1,
    number_starting_markers_skip=0,
    figure_width=16.256,
    axis_font_size=8,
):

    plt.subplots(figsize=(figure_width * cm, figure_width / aspect_ratio * cm))

    plt.rcParams.update({"font.size": fontsize})
    # plt.figure(figsize=(cm_to_inch(12),cm_to_inch(14)))
    size_marker = 7
    if colorfills is None:
        colorfills = colors
    for i in range(len(list_data)):
        if list_legend != []:
            if log_x and log_y:
                plt.loglog(
                    list_x[i],
                    list_data[i],
                    colors[i],
                    markerfacecolor=colorfills[i],
                    markeredgewidth=linewidth_markers,
                    marker=markers[i],
                    markersize=size_marker,
                    markevery=np.logspace(0, np.log10(len(list_data[i]) - 2), 10)[
                        number_starting_markers_skip:
                    ]
                    .astype(int)
                    .tolist(),
                    linewidth=linewidth_figures,
                    label=list_legend[i],
                )
            if log_x and not log_y:
                plt.semilogx(
                    list_x[i],
                    list_data[i],
                    colors[i],
                    markerfacecolor=colorfills[i],
                    markeredgewidth=linewidth_markers,
                    marker=markers[i],
                    markersize=size_marker,
                    linewidth=linewidth_figures,
                    label=list_legend[i],
                )
            if not log_x and log_y:
                plt.semilogy(
                    list_x[i],
                    list_data[i],
                    colors[i],
                    markerfacecolor=colorfills[i],
                    markeredgewidth=linewidth_markers,
                    marker=markers[i],
                    markersize=size_marker,
                    markevery=np.linspace(
                        0, len(list_data[i]) - 2, 10, dtype=int
                    ).tolist()[number_starting_markers_skip:],
                    linewidth=linewidth_figures,
                    label=list_legend[i],
                )
            if not log_x and not log_y:
                plt.plot(
                    list_x[i],
                    list_data[i],
                    colors[i],
                    markerfacecolor=colorfills[i],
                    markeredgewidth=linewidth_markers,
                    marker=markers[i],
                    markersize=size_marker,
                    markevery=np.linspace(
                        0, len(list_data[i]) - 2, 10, dtype=int
                    ).tolist()[number_starting_markers_skip:],
                    linewidth=linewidth_figures,
                    label=list_legend[i],
                )
        else:
            if log_x and log_y:
                plt.loglog(
                    list_x[i],
                    list_data[i],
                    colors[i],
                    marker=markers[i],
                    markerfacecolor=colorfills[i],
                    markeredgewidth=linewidth_markers,
                    markersize=size_marker,
                    markevery=np.logspace(0, np.log10(len(list_data[i]) - 2), 10)
                    .astype(int)
                    .tolist()[number_starting_markers_skip:],
                    linewidth=linewidth_figures,
                )
            if log_x and not log_y:
                plt.semilogx(
                    list_x[i],
                    list_data[i],
                    colors[i],
                    marker=markers[i],
                    markeredgewidth=linewidth_markers,
                    markerfacecolor=colorfills[i],
                    markersize=size_marker,
                    linewidth=linewidth_figures,
                )
            if not log_x and log_y:
                plt.semilogy(
                    list_x[i],
                    list_data[i],
                    colors[i],
                    marker=markers[i],
                    markeredgewidth=linewidth_markers,
                    markerfacecolor=colorfills[i],
                    markersize=size_marker,
                    markevery=np.linspace(
                        0, len(list_data[i]) - 2, 10, dtype=int
                    ).tolist()[number_starting_markers_skip:],
                    linewidth=linewidth_figures,
                )
            if not log_x and not log_y:
                plt.plot(
                    list_x[i],
                    list_data[i],
                    colors[i],
                    marker=markers[i],
                    markeredgewidth=linewidth_markers,
                    markerfacecolor=colorfills[i],
                    markersize=size_marker,
                    markevery=np.linspace(
                        0, len(list_data[i]) - 2, 10, dtype=int
                    ).tolist()[number_starting_markers_skip:],
                    linewidth=linewidth_figures,
                )
    plt.title(title, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.xticks(fontsize=axis_font_size)
    plt.yticks(fontsize=axis_font_size)

    # ax = plt.gca()
    # ax.set_yticklabels([])
    # ax.set_yticks([])

    if x_limits is not None:
        plt.xlim(x_limits)
    if y_limits is not None:
        plt.ylim(y_limits)
    if list_legend != []:
        if legend_location is not None:
            legend = plt.legend(
                fontsize=label_font_size,
                loc=legend_location,
                ncol=number_columns_legend,
                framealpha=framealpha_val,
                labelspacing=label_spacing_val,
                borderpad=borderpad_val,
            )
        else:
            if outside_legend:
                legend = plt.legend(
                    fontsize=label_font_size,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=number_columns_legend,
                    framealpha=framealpha_val,
                    labelspacing=label_spacing_val,
                    borderpad=borderpad_val,
                )
            else:
                legend = plt.legend(
                    fontsize=label_font_size,
                    ncol=number_columns_legend,
                    framealpha=framealpha_val,
                    labelspacing=label_spacing_val,
                    borderpad=borderpad_val,
                )
    plt.tight_layout()
    plt.minorticks_off()
    plt.grid(True)
    if save_figure is None:
        plt.show()
    else:
        if ".pdf" in save_figure:
            plt.savefig(save_figure, bbox_inches="tight", transparent=True)
        if ".eps" in save_figure:
            plt.savefig(
                save_figure, bbox_inches="tight", format="eps", transparent=True
            )
        if ".png" in save_figure:
            plt.savefig(
                save_figure,
                dpi=600,
                format="png",
                bbox_inches="tight",
                transparent=True,
            )
        # plt.savefig(save_figure)
        plt.close()


linewidth_figures_shaded = 2.0
shading_figures = 0.3


def plot_results_shaded(
    list_x,
    list_std_dev,
    list_data,
    list_legend,
    title,
    x_label,
    y_label,
    colors,
    markers,
    colorfills=None,
    save_figure=None,
    legend_location=None,
    outside_legend=False,
    x_limits=None,
    y_limits=None,
    label_font_size=8,
    number_columns_legend=1,
    figure_width=16.256,
    axis_font_size=8,
):

    plt.subplots(figsize=(figure_width * cm, figure_width / aspect_ratio * cm))
    plt.rcParams.update({"font.size": fontsize})
    # plt.figure(figsize=(cm_to_inch(12),cm_to_inch(14)))
    size_marker = 7
    if colorfills is None:
        colorfills = colors
    for i in range(len(list_data)):
        plt.fill_between(
            list_x[i],
            list_data[i] - list_std_dev[i],
            list_data[i] + list_std_dev[i],
            color=colors[i],
            alpha=shading_figures,
        )
        if list_legend != []:
            plt.plot(
                list_x[i],
                list_data[i],
                colors[i],
                markerfacecolor=colorfills[i],
                markeredgewidth=linewidth_markers,
                marker=markers[i],
                markersize=size_marker,
                markevery=np.linspace(0, len(list_data[i]) - 2, 10, dtype=int).tolist(),
                linewidth=linewidth_figures_shaded,
                label=list_legend[i],
            )
        else:
            plt.plot(
                list_x[i],
                list_data[i],
                colors[i],
                marker=markers[i],
                markeredgewidth=linewidth_markers,
                markerfacecolor=colorfills[i],
                markersize=size_marker,
                markevery=np.linspace(0, len(list_data[i]) - 2, 10, dtype=int).tolist(),
                linewidth=linewidth_figures_shaded,
            )
    plt.title(title, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    if x_limits is not None:
        plt.xlim(x_limits)
    if y_limits is not None:
        plt.ylim(y_limits)
    if list_legend != []:
        if legend_location is not None:
            legend = plt.legend(
                fontsize=label_font_size,
                loc=legend_location,
                ncol=number_columns_legend,
                framealpha=framealpha_val,
                labelspacing=label_spacing_val,
                borderpad=borderpad_val,
            )
        else:
            if outside_legend:
                legend = plt.legend(
                    fontsize=label_font_size,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=number_columns_legend,
                    framealpha=framealpha_val,
                    labelspacing=label_spacing_val,
                    borderpad=borderpad_val,
                )
            else:
                legend = plt.legend(
                    fontsize=label_font_size,
                    ncol=number_columns_legend,
                    framealpha=framealpha_val,
                    labelspacing=label_spacing_val,
                    borderpad=borderpad_val,
                )
    plt.tight_layout()
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.xticks(fontsize=axis_font_size)
    plt.yticks(fontsize=axis_font_size)
    plt.minorticks_off()
    plt.grid(True)
    if save_figure is None:
        plt.show()
    else:
        if ".pdf" in save_figure:
            plt.savefig(save_figure, bbox_inches="tight", transparent=True)
        if ".eps" in save_figure:
            plt.savefig(
                save_figure, bbox_inches="tight", format="eps", transparent=True
            )
        if ".png" in save_figure:
            plt.savefig(
                save_figure,
                dpi=600,
                format="png",
                bbox_inches="tight",
                transparent=True,
            )
        # plt.savefig(save_figure)
        plt.close()
