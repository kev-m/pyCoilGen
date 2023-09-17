import numpy as np
from typing import List

import matplotlib.pyplot as plt

from pyCoilGen.sub_functions.data_structures import CoilSolution, SolutionErrors, FieldErrors, TargetField


def plot_vector_field_xy(coords: np.ndarray, field: np.ndarray, plot_title='plot_vector_field_xy', save_figure=False):
    """
    Generate a contour plot of the field vectors in the X-Z plane.

    Args:
        coords (np.ndarray): The field co-ordinates (3,n)
        field (np.ndarray): The field vector (3,n)

    Returns:
        None
    """
    # Find the middle z-coordinate value
    middle_z = np.median(coords[2])

    # Define the extent of your data (adjust as needed)
    x_min, x_max = np.min(coords[0]), np.max(coords[0])
    y_min, y_max = np.min(coords[1]), np.max(coords[1])

    # Create a grid of x and y coordinates
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Evaluate the magnetic field strength at each point
    z = np.full_like(x, middle_z)  # Set z to middle z-coordinate
    b_field = np.zeros_like(x)

    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([x[i, j], y[i, j], z[i, j]]).reshape(3, 1)
            dist = np.linalg.norm(coords - point, axis=0)
            weights = 1 / dist
            weights /= np.sum(weights)
            b_field[i, j] = np.sum(field * weights)

    # Create a contour plot
    plt.figure(figsize=(8, 6))
    contours = plt.contourf(x, y, b_field, cmap='viridis')
    plt.colorbar(contours)
    plt.title('Contour Plot of Magnetic Field Strengths\n(X-Y Plane)')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Save the figure if specified
    if save_figure:
        plt.savefig(f'images/{plot_title}.png', dpi=75)
    else:
        plt.show()


def plot_vector_field(coords: np.ndarray, field: np.ndarray, magic=[0, 1, 2], plot_title='plot_vector_field_', save_figure=False):
    """
    Generate a contour plot of the field vectors in a plane.

    Args:
        coords (np.ndarray): The field co-ordinates (3,n)
        field (np.ndarray): The field vector (3,n)
        magic (ints): The indices of the 3 axes.
        plot_title (str): Label to use for plotting.
        save_figure(bool, optional): Whether to save the figure as an image file or to plot it.

    Returns:
        None
    """

    axis_chars = [str(chr(ord('x') + magic_value)) for magic_value in magic]
    if plot_title[-1] == '_':
        plot_title += axis_chars[2]
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
    plt.title(f'Contour Plot of Magnetic Field Strengths\n({axis_chars[2].upper()} Plane = {round(middle_z,3)})')
    plt.xlabel(axis_chars[0].upper())
    plt.ylabel(axis_chars[1].upper())

    # Save the figure if specified
    if save_figure:
        plt.savefig(f'images/{plot_title}.png', dpi=75)
    else:
        plt.show()


def plot_vector_field_xy(coords: np.ndarray, field: np.ndarray, plot_title='plot_vector_field_xy', save_figure=False):
    """
    Generate a contour plot of the field vectors in the X-Y plane, centred on the Z-values.

    Args:
        coords (np.ndarray): The field co-ordinates (3,n)
        field (np.ndarray): The field vector (3,n)
        plot_title (str): Label to use for plotting.
        save_figure(bool, optional): Whether to save the figure as an image file or to plot it.

    Returns:
        None
    """
    return plot_vector_field(coords=coords, field=field, magic=[0, 1, 2], plot_title=plot_title, save_figure=save_figure)


def plot_vector_field_yz(coords: np.ndarray, field: np.ndarray, plot_title='plot_vector_field_yz', save_figure=False):
    """
    Generate a contour plot of the field vectors in the Y-Z plane, centred on the X-values.

    Args:
        coords (np.ndarray): The field co-ordinates (3,n)
        field (np.ndarray): The field vector (3,n)
        plot_title (str): Label to use for plotting.
        save_figure(bool, optional): Whether to save the figure as an image file or to plot it.

    Returns:
        None
    """
    return plot_vector_field(coords=coords, field=field, magic=[1, 2, 0], plot_title=plot_title, save_figure=save_figure)


def plot_vector_field_xz(coords: np.ndarray, field: np.ndarray, plot_title='plot_vector_field_xz', save_figure=False):
    """
    Generate a contour plot of the field vectors in the X-Z plane, centred on Y.

    Args:
        coords (np.ndarray): The field co-ordinates (3,n)
        field (np.ndarray): The field vector (3,n)
        plot_title (str): Label to use for plotting.
        save_figure(bool, optional): Whether to save the figure as an image file or to plot it.

    Returns:
        None
    """
    return plot_vector_field(coords=coords, field=field, magic=[2, 0, 1], plot_title=plot_title, save_figure=save_figure)
