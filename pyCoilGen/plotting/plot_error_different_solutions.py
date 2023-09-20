import numpy as np
from typing import List

import matplotlib.pyplot as plt

from pyCoilGen.sub_functions.data_structures import CoilSolution
from pyCoilGen.helpers.common import title_to_filename


def plot_error_different_solutions(coil_solutions: List[CoilSolution], solutions_to_plot: List[int], plot_title: str,
                                   x_ticks: dict = None, save_dir=None, dpi=100):
    """
    Plots error metrics for different coil solutions.

    If requested, with save_figure, images are saved to an 'images' subdirectory.

    Args:
        coil_solutions (List[CoilSolution]): List of CoilSolution objects.
        solutions_to_plot (List[int]): List of indices indicating which solutions to plot.
        plot_title (str): Title of the plot.
        x_ticks (dict): A single key-values dictionary to specify custom x-ticks. The key is used to label the x-axis.
        save_dir (str, optional): If specified, saves the plot to the directory, else plots it.
        dpi (int, optional): The dots-per-inch (DPI) to use when saving the figure.

    Returns:
        None
    """
    max_rel_error_layout_vs_target = np.array(
        [coil_solutions[x].solution_errors.field_error_vals.max_rel_error_layout_vs_target for x in solutions_to_plot])
    mean_rel_error_layout_vs_target = np.array(
        [coil_solutions[x].solution_errors.field_error_vals.mean_rel_error_layout_vs_target for x in solutions_to_plot])
    max_rel_error_loops_vs_target = np.array(
        [coil_solutions[x].solution_errors.field_error_vals.max_rel_error_unconnected_contours_vs_target for x in solutions_to_plot])
    mean_rel_error_loops_vs_target = np.array(
        [coil_solutions[x].solution_errors.field_error_vals.mean_rel_error_unconnected_contours_vs_target for x in solutions_to_plot])
    max_rel_error_layout_vs_sf = np.array(
        [coil_solutions[x].solution_errors.field_error_vals.max_rel_error_layout_vs_stream_function_field for x in solutions_to_plot])
    mean_rel_error_layout_vs_sf = np.array(
        [coil_solutions[x].solution_errors.field_error_vals.mean_rel_error_layout_vs_stream_function_field for x in solutions_to_plot])
    max_rel_error_loops_vs_sf = np.array(
        [coil_solutions[x].solution_errors.field_error_vals.max_rel_error_unconnected_contours_vs_stream_function_field for x in solutions_to_plot])
    mean_rel_error_loops_vs_sf = np.array(
        [coil_solutions[x].solution_errors.field_error_vals.mean_rel_error_unconnected_contours_vs_stream_function_field for x in solutions_to_plot])

    plt.figure(figsize=(10, 6))

    p1 = plt.plot(solutions_to_plot, max_rel_error_loops_vs_sf, 'o-b', linewidth=2)
    p2 = plt.plot(solutions_to_plot, max_rel_error_layout_vs_sf, 'o-r', linewidth=2)
    p3 = plt.plot(solutions_to_plot, mean_rel_error_loops_vs_sf, '*-b', linewidth=2)
    p4 = plt.plot(solutions_to_plot, mean_rel_error_layout_vs_sf, '*-r', linewidth=2)
    p5 = plt.plot(solutions_to_plot, max_rel_error_loops_vs_target, 'o-y', linewidth=2)
    p6 = plt.plot(solutions_to_plot, max_rel_error_layout_vs_target, 'o-m', linewidth=2)
    p7 = plt.plot(solutions_to_plot, mean_rel_error_loops_vs_target, '*-',
                  linewidth=2, color=[0, 0.5, 0], markerfacecolor=[0, 0.5, 0])
    p8 = plt.plot(solutions_to_plot, mean_rel_error_layout_vs_target, '*-c', linewidth=2)

    plt.legend(['max_rel_error_loops_vs_sf', 'max_rel_error_layout_vs_sf', 'mean_rel_error_loops_vs_sf', 'mean_rel_error_layout_vs_sf',
                'max_rel_error_loops_vs_target', 'max_rel_error_layout_vs_target', 'mean_rel_error_loops_vs_target', 'mean_rel_error_layout_vs_target'])
    if x_ticks is not None:
        x_tick = next(iter(x_ticks.keys()))
        x_labels = [str(tick) for tick in x_ticks[x_tick]]
        plt.xlabel(x_tick)
        plt.xticks(solutions_to_plot, x_labels)
    else:
        plt.xlabel('solutions_to_plot')
    plt.ylabel('Error Values')
    plt.title(plot_title)
    plt.grid(True)
    if save_dir is not None:
        plt.savefig(f'{save_dir}/plot_solutions_{title_to_filename(plot_title)}.png', dpi=dpi)
    else:
        plt.show()
