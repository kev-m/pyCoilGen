
import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import pyvista as pv
from pyCoilGen.plotting.plot_wire_loops import plot_wire_loops, plot_wire_loops_tube


# ==========================================================
# Extract wire path with wire spacing 
# ==========================================================
def extract_wire_paths(
        verts,
        faces,
        psi,
        levels,
        display_debug_plots=False,
        smooth_resample=4000,
        wire_width=0.002,
        wire_thickness=0.0016,
        clearance=0.0005,
        terminal_cut_length=0.0015):
    
    """
    Extract wire paths from a triangulated surface using contour levels.
    This function generates spiral wire paths by extracting contours from a
    triangulated surface, enforcing spacing constraints, and connecting loops
    with smooth arc bridges. It handles both positive and negative polarity
    coils separately.
    Args:
        verts (np.ndarray): Vertex coordinates of shape (n, 3) representing
            the triangulated surface mesh.
        faces (np.ndarray): Face connectivity array of shape (m, 3) defining
            triangulation.
        psi (np.ndarray): Scalar field values at vertices for contour extraction.
        levels (array-like): Contour levels to extract from the scalar field.
        display_debug_plots (bool, optional): If True, display visualization of
            wire loops. Defaults to False.
        smooth_resample (int, optional): Number of points for resampling smoothed
            paths. Defaults to 4000.
        wire_width (float, optional): Width of the wire in meters. Defaults to 0.002.
        wire_thickness (float, optional): Thickness of the wire in meters.
            Defaults to 0.0016.
        clearance (float, optional): Minimum clearance between wires in meters.
            Defaults to 0.0005.
        terminal_cut_length (float, optional): Length to trim from terminal ends
            in meters. Defaults to 0.0015.
    Returns:
        list: List of dictionaries, each containing:
            - "path" (np.ndarray): 3D coordinates of wire centerline.
            - "sign" (int): Coil polarity (1 for positive, -1 for negative).
            - "width" (float): Wire width.
            - "thickness" (float): Wire thickness.
    Raises:
        No explicit exceptions. Gracefully handles edge cases with fallbacks.
    """
    z_plate = np.mean(verts[:, 2])

    # Minimum spacing between wire centerlines
    min_spacing = wire_width + clearance + 0.25 * wire_width

    triang = mtri.Triangulation(verts[:, 0], verts[:, 1], faces)

    fig, ax = plt.subplots()
    cs = ax.tricontour(triang, psi, levels=levels)
    plt.close(fig)

    # ------------------------------------------------
    # Smooth path
    # ------------------------------------------------
    def smooth_path(path):

        if len(path) < 10:
            return path

        try:

            k = min(3, len(path) - 1)

            tck, u = splprep(
                [path[:,0], path[:,1], path[:,2]],
                s=1e-6,
                k=k
            )

            u_new = np.linspace(0,1,smooth_resample)

            x,y,z = splev(u_new,tck)

            return np.column_stack([x,y,z])

        except:
            return path

    # ------------------------------------------------
    # Arc bridge
    # ------------------------------------------------
    def arc_bridge(p1, p2, z, npts=40):

        r1 = np.sqrt(p1[0]**2 + p1[1]**2)
        r2 = np.sqrt(p2[0]**2 + p2[1]**2)

        r = 0.5 * (r1 + r2)

        th1 = np.arctan2(p1[1], p1[0])
        th2 = np.arctan2(p2[1], p2[0])

        dth = th2 - th1

        if dth > np.pi:
            dth -= 2*np.pi
        if dth < -np.pi:
            dth += 2*np.pi

        th = np.linspace(th1, th1 + dth, npts)

        x = r * np.cos(th)
        y = r * np.sin(th)

        return np.column_stack([x, y, np.full(npts, z)])

    # ------------------------------------------------
    # Cut terminal end (NEW)
    # ------------------------------------------------
    def cut_terminal_end(path, cut_length):

        if len(path) < 5:
            return path

        seg_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)

        total = 0.0
        idx = len(path) - 1

        while idx > 1 and total < cut_length:
            total += seg_lengths[idx-1]
            idx -= 1

        return path[:idx]

    # ------------------------------------------------
    # Extract loops
    # ------------------------------------------------
    loops = []

    for level, segs in zip(cs.levels, cs.allsegs):

        for seg in segs:

            if len(seg) < 30:
                continue

            r = np.mean(np.sqrt(seg[:,0]**2 + seg[:,1]**2))

            path = np.column_stack([
                seg[:,0],
                seg[:,1],
                np.full(len(seg), z_plate)
            ])

            loops.append({
                "path": path,
                "sign": 1 if level > 0 else -1,
                "r": r
            })

    # Separate polarities
    pos_loops = sorted([l for l in loops if l["sign"] > 0], key=lambda x: x["r"])
    neg_loops = sorted([l for l in loops if l["sign"] < 0], key=lambda x: x["r"])

    # ------------------------------------------------
    # Enforce global spacing
    # ------------------------------------------------
    def enforce_spacing(loopset):

        accepted = []
        global_pts = None

        for loop in loopset:

            pts = loop["path"]

            if global_pts is None:

                accepted.append(pts)
                global_pts = pts

                continue

            tree = cKDTree(global_pts)

            d,_ = tree.query(pts)

            if np.min(d) > min_spacing:

                accepted.append(pts)
                global_pts = np.vstack([global_pts, pts])

        return accepted

    pos_loops = enforce_spacing(pos_loops)
    neg_loops = enforce_spacing(neg_loops)

    # ------------------------------------------------
    # Build spiral
    # ------------------------------------------------
    def build_spiral(loopset):

        if len(loopset) == 0:
            return None

        spiral = loopset[0].copy()

        cut_pts = max(5, int(0.16 * len(spiral)))
        spiral = spiral[:-cut_pts]

        for loop in loopset[1:]:

            prev_end = spiral[-1]

            tree = cKDTree(loop)

            _, idx = tree.query(prev_end)

            loop = np.roll(loop, -idx, axis=0)

            bridge = arc_bridge(prev_end, loop[0], z_plate)

            spiral = np.vstack([spiral, bridge, loop])

        spiral = smooth_path(spiral)

        # NEW: trim terminal end
        spiral = cut_terminal_end(spiral, terminal_cut_length)

        return spiral

    wire_loops = []

    pos_spiral = build_spiral(pos_loops)
    neg_spiral = build_spiral(neg_loops)

    if pos_spiral is not None:
        wire_loops.append({
            "path": pos_spiral,
            "sign": 1,
            "width": wire_width,
            "thickness": wire_thickness
        })

    if neg_spiral is not None:
        wire_loops.append({
            "path": neg_spiral,
            "sign": -1,
            "width": wire_width,
            "thickness": wire_thickness
        })

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    if display_debug_plots:
       plot_wire_loops_tube(wire_loops, wire_width)

    return wire_loops

