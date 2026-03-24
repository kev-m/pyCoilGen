import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import trimesh
from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
import pyvista as pv

# ----------------------------
# debug plot
# ----------------------------
def plot_wire_loops(wire_loops):
    """
    Plot wire loops for gradient coils.
    Visualizes the paths of wire loops used in gradient coil design,
    with different colors representing positive and negative loops.
    Args:
        wire_loops (list): A list of dictionaries containing wire loop data.
            Each dictionary should have:
            - "path" (numpy.ndarray): An (n, 2) array of (x, y) coordinates
              representing the wire loop path.
            - "sign" (int or float): The polarity of the loop. Positive values
              are plotted in red, non-positive values in blue.
    Returns:
        None
    Raises:
        None
    """

    fig, ax = plt.subplots(figsize=(6,6))
    for loop in wire_loops:
        p = loop["path"]
        ax.plot(p[:,0], p[:,1], 'r' if loop["sign"]>0 else 'b')
    ax.set_aspect("equal")
    ax.set_title("Gradient Coil Wire Paths")
    plt.show()

def plot_wire_loops_tube(wire_loops, wire_width):
    """
    Visualize wire loops as 3D tubes in a plotter.
    Renders wire loops as tubular meshes with color coding based on their sign.
    Positive sign loops are displayed in red, while negative sign loops are
    displayed in blue. The visualization includes axes and grid for reference.
    Args:
        wire_loops (list): A list of dictionaries containing wire loop data.
            Each dictionary must have:
            - "path" (array-like): The 3D coordinates of points defining the loop path.
            - "sign" (float): The sign value determining the tube color (positive
                for red, non-positive for blue).
        wire_width (float): The diameter of the wire tubes in plotting units.
            The tube radius is calculated as wire_width/2.
    Returns:
        None
    Raises:
        None
    Example:
        >>> wire_loops = [
        ...     {"path": [[0, 0, 0], [1, 0, 0], [1, 1, 0]], "sign": 1},
        ...     {"path": [[0, 0, 1], [1, 0, 1], [1, 1, 1]], "sign": -1}
        ... ]
        >>> plot_wire_loops_tube(wire_loops, wire_width=0.1)
    """
        

    plotter = pv.Plotter()

    for loop in wire_loops:

        poly = pv.lines_from_points(loop["path"])

        tube = poly.tube(radius=wire_width/2)

        plotter.add_mesh(
            tube,
            color="red" if loop["sign"] > 0 else "blue"
        )

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


def plot_gerber_paths(gerber_paths, plate_radius_mm, wire_width_mm, part_index):
    """Visualize Gerber paths with wire geometry on a circular plate.
    Plots wire loops extracted from Gerber files, displaying both the path
    centerlines and the actual wire cross-sections with specified width.
    Positive and negative paths are distinguished by color (red and blue).
    Args:
        gerber_paths (list): List of tuples containing (path, sign) pairs where
            path is an Nx2 numpy array of (x, y) coordinates and sign indicates
            the polarity (+1 or -1) of the path.
        plate_radius_mm (float): Radius of the circular plate boundary in
            millimeters. Displayed as a dashed gray circle.
        wire_width_mm (float): Width of the wire in millimeters. Used to
            render rectangular cross-sections along each path segment.
        part_index (int): Index or identifier of the plate being plotted.
            Displayed in the figure title.
    Returns:
        None
    Note:
        - Red paths represent positive polarity (sign > 0)
        - Blue paths represent negative polarity (sign <= 0)
        - Wire cross-sections are rendered as rotated rectangles aligned
          with path segments
        - The plot uses equal aspect ratio for accurate geometric representation
    """
    
    plt.figure(figsize=(6,6))
    ax = plt.gca()

    for path, sign in gerber_paths:

        ax.plot(
            path[:,0],
            path[:,1],
            color="red" if sign>0 else "blue",
            linewidth=1.5
        )

        for i in range(len(path)-1):

            p0, p1 = path[i], path[i+1]

            dx, dy = p1[0]-p0[0], p1[1]-p0[1]

            length = np.sqrt(dx**2 + dy**2)

            angle = np.degrees(np.arctan2(dy, dx))

            

            rect = Rectangle(
                (p0[0]-wire_width_mm/2, p0[1]-wire_width_mm/2),
                length,
                wire_width_mm,
                angle=angle,
                color="red" if sign>0 else "blue",
                alpha=0.2
            )

            ax.add_patch(rect)

    circle = plt.Circle(
        (0,0),
        plate_radius_mm,
        fill=False,
        linestyle='--',
        color='gray'
    )

    ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")

    plt.title(
        f"Plate {part_index} Wire Loops with {wire_width_mm:.1f} mm Width"
    )

    plt.show()

def plot_stl_patch(gerber_paths, plate_radius_mm, wire_width_mm, part_index):
    """
    Plot wire loops with terminals and strain relief on a circular plate.
    Visualizes the layout of wire loops on a circular plate, including the plate
    boundary, wire paths, grooves, terminal holes, and strain relief indicators.
    Wire paths are color-coded by sign (red for positive, blue for negative).
    Args:
        gerber_paths (list of tuples): List of (path, sign) tuples where path is
            a numpy array of shape (n, 2) containing 2D coordinates and sign is
            a numeric value indicating the loop polarity (positive or negative).
        plate_radius_mm (float): Radius of the circular plate in millimeters.
        wire_width_mm (float): Width of the wire in millimeters, used for
            calculating groove dimensions and terminal hole sizes.
        part_index (int): Index or identifier of the plate for display in the
            plot title.
    Returns:
        None: Displays the plot using matplotlib.
    Raises:
        None
    Note:
        - Red wire paths indicate positive polarity, blue indicate negative.
        - Green semi-transparent circles mark the start and end terminal holes.
        - Orange arrows indicate strain relief directions.
        - Groove widths are visualized as semi-transparent rectangles along
          the wire paths.
    """

    plt.figure(figsize=(8,8))
    ax = plt.gca()
    # Draw plate boundary
    circle = Circle(
        (0,0),
        plate_radius_mm,
        fill=False,
        linestyle='--',
        color='gray',
        linewidth=2
    )
    ax.add_patch(circle)

    # Draw wire loops
    for loop_index, (path, sign) in enumerate(gerber_paths):
        ax.plot(
            path[:,0],
            path[:,1],
            color="red" if sign>0 else "blue",
            linewidth=1.5,
            label=f"Loop {loop_index+1}"
        )

        # Draw grooves as semi-transparent rectangles
        for i in range(len(path)-1):
            p0, p1 = path[i], path[i+1]
            dx, dy = p1-p0
            length = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))
            rect = Rectangle(
                (p0[0]-wire_width_mm/2, p0[1]-wire_width_mm/2),
                length,
                wire_width_mm,
                angle=angle,
                color="red" if sign>0 else "blue",
                alpha=0.2
            )
            ax.add_patch(rect)

        # Draw start and end terminal holes
        def tangent(p0, p1):
            v = p1 - p0
            n = np.linalg.norm(v)
            return v/n if n>0 else np.array([1.0,0.0])
        def perp(v): return np.array([-v[1], v[0]])

        t_start = tangent(path[0], path[1])
        t_end = tangent(path[-2], path[-1])
        start_pt = path[0] + perp(t_start)*wire_width_mm*1.5
        end_pt   = path[-1] + perp(t_end)*wire_width_mm*1.5

        ax.add_patch(Circle(start_pt, radius=wire_width_mm*0.8, color='green', alpha=0.5))
        ax.add_patch(Circle(end_pt, radius=wire_width_mm*0.8, color='green', alpha=0.5))

        # Draw strain relief arrows
        strain_len = wire_width_mm*4
        ax.add_patch(FancyArrow(
            start_pt[0], start_pt[1],
            t_start[0]*strain_len, t_start[1]*strain_len,
            width=0.3, color='orange', alpha=0.6
        ))
        ax.add_patch(FancyArrow(
            end_pt[0], end_pt[1],
            t_end[0]*strain_len, t_end[1]*strain_len,
            width=0.3, color='orange', alpha=0.6
        ))

        # Draw loop numbers
        ax.text(path[0][0], path[0][1], str(loop_index+1),
                fontsize=10, color='purple', weight='bold')

        ax.set_aspect('equal')
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        plt.title(f"Plate {part_index} Wire Loops with Terminals and Strain Relief")
        plt.grid(True)
    plt.show()