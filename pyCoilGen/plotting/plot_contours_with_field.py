
import logging
import numpy as np
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pyCoilGen.sub_functions.data_structures import CoilSolution, SolutionErrors
from pyCoilGen.helpers.common import title_to_filename

# Configure logging
log = logging.getLogger(__name__)

_default_colours = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

def plot_contours_with_field(coil_layout: List[CoilSolution], single_ind_to_plot: int, plot_title: str, group_colours=_default_colours, save_dir=None, dpi=100):
    """
    Plot the stream function contours and the calculated field overlayed in a single 3D plot.

    Args:
        coil_layout (List[CoilSolution]): List of CoilSolution objects.
        single_ind_to_plot (int): Index of the solution to plot.
        plot_title (str): Title of the plot.
        group_colours (list of str, optional): Colours for different contour groups.
        save_dir (str, optional): Directory to save the plot. If None, the plot is only displayed.
        dpi (int, optional): Resolution of the saved plot.

    Returns:
        None
    """
    dot_size = 200

    # Extract relevant data from the CoilSolution
    coil_solution: CoilSolution = coil_layout[single_ind_to_plot]
    errors: SolutionErrors = coil_solution.solution_errors

    layout_c = errors.combined_field_layout[2]              # Calculated field
    pos_data = coil_solution.target_field.coords            # Coordinates (3, N)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(plot_title, fontsize=16)

    # Plot the calculated field as a scatter plot
    scatter = ax.scatter(pos_data[0], pos_data[1], pos_data[2], 
                         c=layout_c, s=dot_size, cmap='viridis', label='Calculated Field')

    # Add a color bar for the calculated field
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Field [mT/A]')

    # Plot the contours from the stream function
    for part_ind, coil_part in enumerate(coil_solution.coil_parts):
        normed_sf = coil_part.stream_function - np.min(coil_part.stream_function)
        normed_sf /= np.max(normed_sf)

        # Create vertices and faces for Poly3DCollection
        vertices = coil_part.coil_mesh.v
        faces = coil_part.coil_mesh.f  # Faces as indices
        face_vertices = vertices[faces]

        # Calculate face colors based on normalized stream function
        sf_face_colours = [np.mean(normed_sf[face]) for face in faces]
        face_colors = plt.cm.viridis(sf_face_colours)

        # Plot contour loops if groups are present
        if coil_part.groups is not None:
            for group_ind, group in enumerate(coil_part.groups):
                group_color = group_colours[group_ind % len(group_colours)]
                for contour in group.loops:
                    ax.plot(contour.v[0, :], contour.v[1, :], contour.v[2, :], color=group_color, linewidth=2)

    # Customize plot appearance
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    # Set equal aspect ratio
    combined_mesh = coil_solution.combined_mesh.vertices
    min_values = np.min(combined_mesh, axis=0)
    max_values = np.max(combined_mesh, axis=0)
    ax.set_xlim(min_values[0], max_values[0])
    ax.set_ylim(min_values[1], max_values[1])
    ax.set_zlim(min_values[2], max_values[2])
    plt.gca().set_box_aspect([1, 1, 1])  # Set the aspect ratio to be equal
    plt.tight_layout()

    # Save the plot if save_dir is provided
    if save_dir is not None:
        plt.savefig(f'{save_dir}/plot_contours_with_field_{title_to_filename(plot_title)}.png', dpi=dpi)
        log.info(f'Plot saved to {save_dir}/plot_contours_with_field_{title_to_filename(plot_title)}.png')

    # Display the plot
    plt.show()
