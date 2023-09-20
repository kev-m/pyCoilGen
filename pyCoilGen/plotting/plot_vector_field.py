import numpy as np
from typing import List

import matplotlib.pyplot as plt

from pyCoilGen.sub_functions.data_structures import CoilSolution, SolutionErrors, FieldErrors, TargetField
from pyCoilGen.helpers.common import title_to_filename


def plot_vector_field(coords: np.ndarray, field: np.ndarray, magic=[0, 1, 2], plot_title='Vector Field ', save_dir=None, dpi=100):
    """
    Generate a contour plot of the field vectors in a plane.

    Args:
        coords (np.ndarray): The field co-ordinates (3,n)
        field (np.ndarray): The field vector (3,n)
        magic (ints): The indices of the 3 axes.
        plot_title (str): Label to use for plotting. If it ends with a 'space', the plane will by added automatically.
        save_dir (str, optional): If specified, saves the plot to the directory, else plots it.
        dpi (int, optional): The dots-per-inch (DPI) to use when saving the figure.

    Returns:
        None
    """

    axis_chars = [str(chr(ord('X') + magic_value)) for magic_value in magic]
    if plot_title[-1] == ' ':
        plot_title += axis_chars[0]
        plot_title += axis_chars[1]
    # Find the middle z-coordinate value
    middle_z = np.median(coords[magic[2]])

    # Find the z's that are near the middle
    m_indices = np.where(np.abs(coords[magic[2]] - middle_z) <= 1e-4)[0]
    zzz = coords[:, m_indices]
    xs = np.unique(zzz[magic[0]])
    ys = np.unique(zzz[magic[1]])

    # Initialize z_matrix with zeros
    z_matrix = np.full((len(ys), len(xs)), np.nan)

    # Iterate through m_indices and populate z_matrix
    for idx in m_indices:
        x_idx = np.where(xs == coords[magic[0], idx])[0][0]
        y_idx = np.where(ys == coords[magic[1], idx])[0][0]
        z_matrix[y_idx, x_idx] = field[magic[2], idx]

    # Create a contour plot
    plt.figure(figsize=(8, 6))
    contours = plt.contourf(xs, ys, z_matrix, cmap='viridis')
    plt.colorbar(contours)
    plt.title(f'Plot of {plot_title}\n({axis_chars[2]} Plane = {round(middle_z,3)})')
    plt.xlabel(axis_chars[0])
    plt.ylabel(axis_chars[1])

    # Save the figure if specified
    if save_dir is not None:
        plt.savefig(f'{save_dir}/plot_{title_to_filename(plot_title)}.png', dpi=dpi)
    else:
        plt.show()


def plot_vector_field_xy(coords: np.ndarray, field: np.ndarray, plot_title='Vector Field XY', save_dir=None, dpi=100):
    """
    Generate a contour plot of the field vectors in the X-Y plane, centred on the Z-values.

    Args:
        coords (np.ndarray): The field co-ordinates (3,n)
        field (np.ndarray): The field vector (3,n)
        plot_title (str): Label to use for plotting.
        save_dir (str, optional): If specified, saves the plot to the directory, else plots it.
        dpi (int, optional): The dots-per-inch (DPI) to use when saving the figure.

    Returns:
        None
    """
    return plot_vector_field(coords=coords, field=field, magic=[0, 1, 2], plot_title=plot_title, save_dir=save_dir, dpi=dpi)


def plot_vector_field_yz(coords: np.ndarray, field: np.ndarray, plot_title='Vector Field YZ', save_dir=None, dpi=100):
    """
    Generate a contour plot of the field vectors in the Y-Z plane, centred on the X-values.

    Args:
        coords (np.ndarray): The field co-ordinates (3,n)
        field (np.ndarray): The field vector (3,n)
        plot_title (str): Label to use for plotting.
        save_dir (str, optional): If specified, saves the plot to the directory, else plots it.
        dpi (int, optional): The dots-per-inch (DPI) to use when saving the figure.

    Returns:
        None
    """
    return plot_vector_field(coords=coords, field=field, magic=[1, 2, 0], plot_title=plot_title, save_dir=save_dir, dpi=dpi)


def plot_vector_field_xz(coords: np.ndarray, field: np.ndarray, plot_title='Vector Field XZ', save_dir=None, dpi=100):
    """
    Generate a contour plot of the field vectors in the X-Z plane, centred on Y.

    Args:
        coords (np.ndarray): The field co-ordinates (3,n)
        field (np.ndarray): The field vector (3,n)
        plot_title (str): Label to use for plotting.
        save_dir (str, optional): If specified, saves the plot to the directory, else plots it.
        dpi (int, optional): The dots-per-inch (DPI) to use when saving the figure.

    Returns:
        None
    """
    return plot_vector_field(coords=coords, field=field, magic=[2, 0, 1], plot_title=plot_title, save_dir=save_dir, dpi=dpi)
