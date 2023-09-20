import logging
import numpy as np
from typing import List

import matplotlib.pyplot as plt

from pyCoilGen.sub_functions.data_structures import CoilSolution, SolutionErrors, FieldErrors
from pyCoilGen.helpers.common import title_to_filename


def plot_various_error_metrics(coil_layouts: List[CoilSolution], single_ind_to_plot: int, plot_title: str, save_dir=None, dpi=100):
    """
    Plots various error metrics for a specific coil layout.

    If requested, with save_figure, images are saved to an 'images' subdirectory.

    Args:
        coil_layouts (List[CoilSolution]): List of coil solutions.
        single_ind_to_plot (int): Index of the coil layout to be plotted.
        plot_title (str): Title of the plot.
        save_dir (str, optional): If specified, saves the plot to the directory, else plots it.
        dpi (int, optional): The dots-per-inch (DPI) to use when saving the figure.

    Returns:
        None
    """
    dot_size = 200

    coil_solution: CoilSolution = coil_layouts[single_ind_to_plot]
    errors: SolutionErrors = coil_solution.solution_errors

    layout_c = errors.combined_field_layout[2]              # field_by_layout (3,257)
    sf_c = coil_solution.sf_b_field[:, 2]                    # b_field_opt_sf (257,3)
    loops_c = errors.combined_field_loops[2]                # field_by_unconnected_loops (3,257)
    target_c = coil_solution.target_field.b[2]              # target_field.b (3,257)
    # loops_c_1A = errors.combined_field_loops_per1Amp[2]     # field_loops_per1Amp (3,257)
    # layout_c_1A = errors.combined_field_layout_per1Amp[2]   #  field_layout_per1Amp (3,257)
    pos_data = coil_solution.target_field.coords            # (3,257)

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(plot_title, fontsize=16)

    # Create subplots
    ax1 = fig.add_subplot(331, projection='3d')
    ax2 = fig.add_subplot(332, projection='3d')
    ax3 = fig.add_subplot(333, projection='3d')
    ax4 = fig.add_subplot(334, projection='3d')
    ax5 = fig.add_subplot(335, projection='3d')
    ax6 = fig.add_subplot(336, projection='3d')
    ax7 = fig.add_subplot(337, projection='3d')
    ax8 = fig.add_subplot(338, projection='3d')
    ax9 = fig.add_subplot(339, projection='3d')

    plot_data = [
        (ax1, target_c, 'Target Bz, [mT/A]'),
        (ax2, sf_c, 'Bz by stream function, [mT/A]'),
        (ax3, layout_c, 'Layout Bz, [mT/A]'),
        (ax4, loops_c, 'Unconnected Contour Bz, [mT/A]'),
        (ax5, abs(sf_c - target_c) / np.max(np.abs(target_c)) * 100, 'Relative SF error, [%]'),
        (ax6, abs(layout_c - target_c) / np.max(np.abs(target_c)) * 100, 'Relative error\n layout vs. target, [%]'),
        (ax7, abs(layout_c - sf_c) / np.max(np.abs(sf_c)) * 100, 'Relative error\n layout vs. sf field, [%]'),
        (ax8, abs(loops_c - target_c) / np.max(np.abs(target_c)) *
         100, 'Relative error\n unconnected contours vs. target, [%]'),
        (ax9, abs(loops_c - layout_c) / np.max(np.abs(target_c)) * 100,
         'Field difference between unconnected contours\n and final layout, [%]')
    ]

    for ax, plot_color, title in plot_data:
        ax.scatter(pos_data[0], pos_data[1], pos_data[2], c=plot_color, s=dot_size, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('x[m]')
        ax.set_ylabel('y[m]')
        ax.set_zlabel('z[m]')
        fig.colorbar(ax.scatter(pos_data[0], pos_data[1], pos_data[2], c=plot_color,
                     s=dot_size, cmap='viridis'), ax=ax, label='Error %', pad=0.10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir is not None:
        plt.savefig(f'{save_dir}/plot_errors_{title_to_filename(plot_title)}.png', dpi=dpi)
    else:
        plt.show()
