import numpy as np
from typing import List

import matplotlib.pyplot as plt

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilSolution, CoilPart
from pyCoilGen.helpers.common import title_to_filename

log = logging.getLogger(__name__)

_default_colours = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']


def plot_2D_contours_with_sf(coil_layout: List[CoilSolution], single_ind_to_plot: int, plot_title: str, group_colours=_default_colours, save_dir=None, dpi=100):
    """
    Plot a single solution with all steps.

    If requested, with save_figure, images are saved to an 'images' subdirectory.

    Args:
        coil_layout (list[CoilSolution]): List of CoilSolution objects.
        single_ind_to_plot (int): Index of the solution to plot.
        plot_title (str): Title of the plot.
        group_colours (list of colour strings, optional): A list of colours to use when plotting group contours.
        save_dir (str, optional): If specified, saves the plot to the directory, else plots it.
        dpi (int, optional): The dots-per-inch (DPI) to use when saving the figure.

    Returns:
        None
    """

    # Extract commonly accessed variables
    coil_solution = coil_layout[single_ind_to_plot]

    # Plot a single solution with all steps
    if coil_solution.coil_parts[0].groups is not None:  # Attribute is always present, but not always initialised
        # If 'groups' attribute is present in coil_parts
        # Create a figure with tiled layout
        num_parts = len(coil_solution.coil_parts)

        if num_parts == 1:
            fig, axs = plt.subplots(1, num_parts, figsize=(6, 5))
            axs = [axs]
        else:
            fig, axs = plt.subplots(1, num_parts, figsize=(6*num_parts, 5))

        for part_ind in range(num_parts):
            coil_part: CoilPart = coil_solution.coil_parts[part_ind]
            ax_part = axs[part_ind]
            ax_part.set_title(f"{plot_title}\nSF Part{part_ind + 1}")

            # Plot the ungrouped and unconnected contour lines with the potential value
            pcolormesh = ax_part.tripcolor(
                coil_solution.coil_parts[part_ind].coil_mesh.uv[:, 0],
                coil_solution.coil_parts[part_ind].coil_mesh.uv[:, 1],
                coil_solution.coil_parts[part_ind].stream_function,
                shading='flat', cmap='viridis'
            )

            # Add a colorbar for the stream function values
            fig.colorbar(pcolormesh, ax=ax_part)

            # Loop through groups and loops within each group
            group_index = 0
            for group in coil_part.groups:
                for loop in group.loops:
                    ax_part.plot(
                        loop.uv[0],
                        loop.uv[1],
                        '-o',
                        linewidth=1,
                        markersize=0.5,
                        color=group_colours[group_index % len(group_colours)]  # Cycle through the default colours
                    )
                group_index += 1

            ax_part.set_yticks([])
            ax_part.set_xticks([])
            ax_part.set_aspect('equal')
            ax_part.set_facecolor('white')

    # Attribute is always present, but not always initialised
    elif coil_solution.coil_parts[0].contour_lines is not None:
        # If 'contour_lines' attribute is present in coil_parts
        # Create a figure with tiled layout
        num_parts = len(coil_solution.coil_parts)
        if num_parts == 1:
            fig, axs = plt.subplots(1, num_parts, figsize=(5, 5))
            axs = [axs]
        else:
            fig, axs = plt.subplots(1, num_parts, figsize=(5*num_parts, 5))

        for part_ind in range(num_parts):
            coil_part = coil_solution.coil_parts[part_ind]
            ax_part = axs[part_ind]
            ax_part.set_title(f"{plot_title}\nSF Part{part_ind + 1}", interpreter='none')

            pcolormesh = ax_part.tripcolor(
                coil_solution.coil_parts[part_ind].coil_mesh.uv[:, 0],
                coil_solution.coil_parts[part_ind].coil_mesh.uv[:, 1],
                coil_solution.coil_parts[part_ind].stream_function,
                shading='flat', cmap='viridis'
            )

            # Add a colorbar for the stream function values
            fig.colorbar(pcolormesh, ax=ax_part)

            # Loop through contour lines
            for contour_line in coil_part.contour_lines:
                ax_part.plot(
                    contour_line.uv[0],
                    contour_line.uv[1],
                    'k-o',
                    linewidth=1,
                    markersize=0.5
                )

            ax_part.set_yticks([])
            ax_part.set_xticks([])
            ax_part.set_aspect('equal')
            ax_part.set_facecolor('white')

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(f'{save_dir}/plot_{title_to_filename(plot_title)}.png', dpi=dpi)
    else:
        plt.show()
