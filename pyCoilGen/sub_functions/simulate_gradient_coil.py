import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from pyCoilGen.plotting.plot_field import plot_field
import pyCoilGen.plotting.plot_wire_loops as plot_wire_loops

# ==========================================================
# Simulate using magpylib
# ==========================================================
def simulate_gradient_coil(coil_parts, DSV_IMAGING, wire, display_field=False):
    """
    Simulate gradient coil B-field including wire width and thickness.
    
    Parameters
    ----------
    coil_parts : list
        List of coil parts, each having `wire_loops` extracted from extract_wire_paths.
    DSV_IMAGING : float
        Diameter of the imaging sphere in meters.
    wire : dict
        Contains {"current": <float>} in Amperes.
    display_field : bool
        If True, display 3D scatter plots of Bx, By, Bz.
    
    Returns
    -------
    dict
        {"points": points, "B": B, "Bx": Bx, "By": By, "Bz": Bz}
    """


    # -----------------------------
    # Create imaging sphere points
    # -----------------------------
    r = DSV_IMAGING / 2
    grid = np.linspace(-r, r, 21)
    points = np.array([[x, y, z]
                       for x in grid for y in grid for z in grid
                       if x**2 + y**2 + z**2 <= r**2])

    current_scale = wire["current"]
    B = np.zeros((len(points), 3))

    # -----------------------------
    # Loop over coil parts and loops
    # -----------------------------
    for part in coil_parts:
        for spiral in part.wire_loops:
            path = spiral["path"]
            width = spiral.get("width", wire['width']) # default to thickness if width not specified
            thickness = spiral.get("thickness", wire['thickness']) # default to width if thickness not specified

            # Downsample if spline too long (memory efficiency)
            if len(path) > 400:
                idx = np.linspace(0, len(path)-1, 400).astype(int)
                path = path[idx]

            # 4-point rectangular quadrature to approximate wire cross-section
            offsets = [(-width/2, -thickness/2),
                       (-width/2,  thickness/2),
                       ( width/2, -thickness/2),
                       ( width/2,  thickness/2)]

            for dx, dz in offsets:
                offset_path = path.copy()
                offset_path[:, 0] += dx
                offset_path[:, 2] += dz

                # Polyline source (current direction taken from vertex order)
                source = magpy.current.Polyline(
                    current=current_scale / len(offsets),  # divided among offsets
                    vertices=offset_path
                )

                # accumulate B-field
                B += magpy.getB(source, points)

    # -----------------------------
    # Extract components
    # -----------------------------
    Bx = B[:, 0]
    By = B[:, 1]
    Bz = B[:, 2]

    # -----------------------------
    # Optional visualization
    # -----------------------------
    if display_field:
       plot_field(Bx, By, Bz, points)

    return {"points": points, "B": B, "Bx": Bx, "By": By, "Bz": Bz}
