import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Import Poly3DCollection
from typing import List

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilSolution

log = logging.getLogger(__name__)

_default_colours = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']


def plot_3D_contours_with_sf(coil_layout: List[CoilSolution], single_ind_to_plot: int, plot_title: str, save_figure=False, group_colours=_default_colours):
    """
    Plot the stream function interpolated on a triangular mesh.

    Args:
        coil_layout (list[CoilSolution]): List of CoilSolution objects.
        single_ind_to_plot (int): Index of the solution to plot.
        plot_title (str): Title of the plot.
        save_figure (bool, optional): Whether to save the figure. Default is False.
        group_colours (list, optional): List of colours for different groups. Default is predefined colours.

    Returns:
        None
    """
    # Plot the stream function interpolated on triangular mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(plot_title + ": " + 'Stream function by optimization and target Bz')

    for part_ind in range(len(coil_layout[single_ind_to_plot].coil_parts)):
        coil_part = coil_layout[single_ind_to_plot].coil_parts[part_ind]
        normed_sf = coil_part.stream_function - np.min(coil_part.stream_function)
        normed_sf /= np.max(normed_sf)

        # Create a list of vertices and faces for Poly3DCollection
        vertices = coil_part.coil_mesh.v
        faces = coil_part.coil_mesh.f  # Get all faces
        face_vertices = vertices[faces]

        # Create a custom colormap using 'viridis'
        sf_face_colours = [np.mean(normed_sf[face]) for face in faces]
        face_colors = plt.cm.viridis(sf_face_colours)

        # Create Poly3DCollection object and add it to the plot
        poly = Poly3DCollection(face_vertices, facecolors=face_colors, alpha=0.6)
        ax.add_collection3d(poly)

        if hasattr(coil_part, 'groups'):
            group_ind = 0
            for group in coil_part.groups:
                group_color = group_colours[group_ind % len(group_colours)]
                for contour in group.loops:
                    ax.plot(contour.v[0, :], contour.v[1, :], contour.v[2, :], color=group_color, linewidth=2)
                group_ind += 1

    # Customize the plot further if needed
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    ax.set_zlabel('z[m]')
    plt.gca().set_box_aspect([1, 1, 1])  # Set the aspect ratio to be equal

    # Save the figure if specified
    if save_figure:
        plt.savefig(f'images/{plot_title}.png', dpi=75)
    else:
        plt.show()
