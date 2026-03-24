import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Point
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
############################################################
# PARAMETERS
############################################################

N_LEVELS = 12
WIRE_RADIUS = 0.001      # 1 mm channel radius (~AWG14)
SMOOTH_SIGMA = 1.0

############################################################
# 1 EXTRACT STREAM FUNCTION
############################################################

def extract_stream_function(coil_parts):

    meshes = []
    psis = []

    for part in coil_parts:
        meshes.append(part.coil_mesh)
        psis.append(part.stream_function)



    return meshes, psis


############################################################
# 2 NORMALIZE STREAM FUNCTION
############################################################

def normalize_stream_function(psi):

    psi = psi - np.mean(psi)
    psi = psi / np.max(np.abs(psi))

    return psi


############################################################
# 3 ROBUST CONTOUR LEVELS
############################################################

def compute_levels(psi, n_levels=N_LEVELS):

    levels = np.quantile(
        psi,
        np.linspace(0.05, 0.95, n_levels)
    )

    return levels


############################################################
# 4 TRIANGLE CONTOUR INTERSECTION
############################################################

def triangle_contour(vertices, psi_vals, level):

    edges = [(0,1),(1,2),(2,0)]
    points = []

    for i,j in edges:

        p1,p2 = psi_vals[i], psi_vals[j]

        if (p1-level)*(p2-level) < 0:

            t = (level-p1)/(p2-p1)

            point = vertices[i] + t*(vertices[j]-vertices[i])

            points.append(point)

    if len(points) == 2:
        return points

    return None


############################################################
# 5 COMPUTE ALL CONTOUR SEGMENTS
############################################################



# ---------------------------------------------------------
# Helper: compute intersection of edge with contour level
# ---------------------------------------------------------

def edge_intersection(v1, v2, p1, p2, level):

    if (p1 - level) * (p2 - level) > 0:
        return None

    if abs(p1 - p2) < 1e-12:
        return None

    t = (level - p1) / (p2 - p1)

    if t < 0 or t > 1:
        return None

    return v1 + t * (v2 - v1)


# ---------------------------------------------------------
# Marching triangles
# ---------------------------------------------------------

def triangle_contour(vertices, psi_vals, level):

    v0, v1, v2 = vertices
    p0, p1, p2 = psi_vals

    pts = []

    e = edge_intersection(v0, v1, p0, p1, level)
    if e is not None:
        pts.append(e)

    e = edge_intersection(v1, v2, p1, p2, level)
    if e is not None:
        pts.append(e)

    e = edge_intersection(v2, v0, p2, p0, level)
    if e is not None:
        pts.append(e)

    if len(pts) == 2:
        return np.array(pts)

    return None


# ---------------------------------------------------------
# Segment stitching
# ---------------------------------------------------------

def stitch_segments(segments, tol=20e-3):

    segments = [tuple(map(tuple, s)) for s in segments]

    unused = set(range(len(segments)))
    loops = []

    while unused:

        idx = unused.pop()
        seg = segments[idx]

        loop = [np.array(seg[0]), np.array(seg[1])]

        growing = True

        while growing:
            growing = False

            for j in list(unused):

                s = segments[j]

                p1 = np.array(s[0])
                p2 = np.array(s[1])

                if np.linalg.norm(loop[-1] - p1) < tol:
                    loop.append(p2)
                    unused.remove(j)
                    growing = True
                    break

                if np.linalg.norm(loop[-1] - p2) < tol:
                    loop.append(p1)
                    unused.remove(j)
                    growing = True
                    break

                if np.linalg.norm(loop[0] - p1) < tol:
                    loop.insert(0, p2)
                    unused.remove(j)
                    growing = True
                    break

                if np.linalg.norm(loop[0] - p2) < tol:
                    loop.insert(0, p1)
                    unused.remove(j)
                    growing = True
                    break

        loops.append(np.array(loop))
        print(f"Formed loop with {len(loop)} points, remaining segments: {len(unused)}")

    return loops


# ---------------------------------------------------------
# Main contour extraction
# ---------------------------------------------------------

def compute_streamfunction_loops(mesh, psi, levels, display_loops:bool = True):

    vertices = mesh.v
    faces = mesh.f

    segments_by_level = defaultdict(list)

    for face in faces:

        tri_v = vertices[face]
        tri_p = psi[face]

        for level in levels:

            seg = triangle_contour(tri_v, tri_p, level)

            if seg is not None:
                segments_by_level[level].append(seg)

    all_loops = []

    for level, segs in segments_by_level.items():

        print("Level", level, "segments:", len(segs))

        loops = stitch_segments(segs)

        print("Level", level, "loops:", len(loops))

        # remove tiny fragments
        loops = [l for l in loops if len(l) > 10]

        all_loops.extend(loops)

    print("Total loops:", len(all_loops))

    if display_loops:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for loop in loops:
            ax.plot(loop[:,0], loop[:,1], loop[:,2], 'r')

        ax.set_title("Extracted Gradient Coil Loops")

        plt.show()

    return all_loops




############################################################
# 6 SMOOTH LOOPS
############################################################

def smooth_loops(loops):

    smoothed = []

    for loop in loops:

        if len(loop) < 5:
            smoothed.append(loop)
            continue

        sm = gaussian_filter1d(loop, SMOOTH_SIGMA, axis=0)

        smoothed.append(sm)

    return smoothed


############################################################
# 8 VISUALIZE LOOPS
############################################################

def visualize_loops(loops):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for loop in loops:

        ax.plot(loop[:,0], loop[:,1], loop[:,2])

    ax.set_title("Gradient Coil Wire Patterns")

    plt.show()


############################################################
# 9 CREATE PRINTABLE WIRE CHANNELS
############################################################

def create_wire_channels(loops, radius=WIRE_RADIUS):

    tubes = []
    profile = Point(0, 0).buffer(radius)

    for loop in loops:

        tube = trimesh.creation.sweep_polygon(
            polygon=profile,
            path=loop,
            cap=False  # Optional: caps the ends
        )

        tubes.append(tube)

    return trimesh.util.concatenate(tubes)

#############################################################
# Visualize the stream function values on the mesh surface
#############################################################

def plot_stream_function(meshes, psis, cmap='coolwarm'):
    fig = plt.figure(figsize=(12, 6))
    
    for i, (mesh, psi) in enumerate(zip(meshes, psis)):
        ax = fig.add_subplot(1, len(meshes), i+1, projection='3d')
        
        # Get vertices and faces
        verts = mesh.v
        faces = mesh.f
        
        # Create a triangulated surface
        tri_collection = ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2],
                                         triangles=faces,
                                         cmap=cmap,
                                         linewidth=0.2,
                                         antialiased=True,
                                         shade=True)
        
        # Map stream function to color
        tri_collection.set_array(psi)
        tri_collection.autoscale()
        
        ax.set_title(f'Coil part {i+1}')
        ax.set_axis_off()
        
    fig.colorbar(tri_collection, ax=fig.axes, shrink=0.5, label='Stream Function')
    plt.show()



############################################################
# 10 COMPLETE PIPELINE
############################################################

def generate_coil_from_stream_function(coil_parts, n_levels, display=True):

    meshes, psis = extract_stream_function(coil_parts)

    if display:
        plot_stream_function(meshes, psis)

    all_loops = []

    for mesh, psi in zip(meshes, psis):
 
        print("Processing mesh with", len(mesh.v), "vertices")

        psi = normalize_stream_function(psi)

        levels = compute_levels(psi, n_levels)

        loops = compute_streamfunction_loops(mesh, psi, levels)

        loops = smooth_loops(loops)

        all_loops.extend(loops)

    print("Total loops:", len(all_loops))

    visualize_loops(all_loops)

    channels = create_wire_channels(all_loops)

    channels.export("gradient_wire_channels.stl")

    print("Exported gradient_wire_channels.stl")

    return all_loops, channels


