import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_field(Bx, By, Bz, points):
    """Visualize magnetic field components in 3D space.
    Creates a figure with three subplots displaying the x, y, and z components
    of a magnetic field. Each component is visualized as a 3D scatter plot with
    color mapping representing the field magnitude at each point.
    Args:
        Bx (array-like): Magnetic field component in the x-direction.
        By (array-like): Magnetic field component in the y-direction.
        Bz (array-like): Magnetic field component in the z-direction.
        points (ndarray): Array of shape (N, 3) containing the 3D coordinates
            of points where the field is sampled, with columns representing
            x, y, and z coordinates respectively.
    Returns:
        None
    Raises:
        None
    Note:
        The function displays the plot using matplotlib's interactive backend.
        All three field components must have the same length as the number of rows
        in the points array.
    """

    fig = plt.figure(figsize=(15,5))
    for i, (comp, title) in enumerate(zip([Bx, By, Bz], ["Bx","By","Bz"])):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        sc = ax.scatter(points[:,0], points[:,1], points[:,2], c=comp, cmap='jet')
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, shrink=0.6)
        fig.suptitle("Simulated Gradient Field")
        plt.tight_layout()
    plt.show()