# System imports
from typing import List
import numpy as np

# Logging
import logging

# Local imports
from sub_functions.data_structures import Mesh
from sub_functions.mesh_parameterization_iterative import mesh_parameterization_iterative
from sub_functions.calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector

# Debugging
# from helpers.visualisation import visualize_vertex_connections

log = logging.getLogger(__name__)


def parameterize_mesh(coil_parts: List[Mesh], input) -> List[Mesh]:
    """
    Create the parameterized 2D mesh.

    Args:
        coil_parts (object): Coil parts object with attributes 'coil_mesh'.
        input (object): Input object with attributes 'surface_is_cylinder_flag' and 'circular_diameter_factor_cylinder_parameterization'.

    Returns:
        object: Updated coil parts object with parameterized mesh.
    """

    # The non-cylindrical parameterization is taken from "matlabmesh @ Ryan Schmidt  rms@dgp.toronto.edu"
    # based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes"

    surface_is_cylinder = input.surface_is_cylinder_flag
    circular_factor = input.circular_diameter_factor

    for part_ind in range(len(coil_parts)):
        mesh_part = coil_parts[part_ind].coil_mesh
        mesh_vertices = mesh_part.get_vertices()
        mesh_faces = mesh_part.get_faces()

        log.debug(" - processing %d, vertices shape: %s", part_ind, mesh_part.get_vertices().shape)

        # Compute face and vertex normals
        face_normals = mesh_part.face_normals()
        vertex_normals = mesh_part.vertex_normals()

        max_face_normal = [np.std(face_normals[:, 0]),
                           np.std(face_normals[:, 1]),
                           np.std(face_normals[:, 2])]
        max_face_normal_std = np.max(max_face_normal)

        log.debug(" - max_face_normal: %s, max_face_normal_std: %s", max_face_normal, max_face_normal_std)

        mesh_part.v = mesh_vertices
        mesh_part.fn = face_normals
        mesh_part.n = vertex_normals

        # Check if vertex coordinates are rather constant in one of the three dimensions
        if not (max_face_normal_std < 1e-6):
            # Go for the parameterization; distinguish between cylinder and non-cylinder
            if not surface_is_cylinder:
                mesh_part = mesh_parameterization_iterative(mesh_part)
                # Create a 2D dataset for fit
            else:
                # Planarization of cylinder
                # DEBUG
                # visualize_vertex_connections(mesh_vertices, 800, 'images/cylinder_projected1.png')

                # Create 2D mesh for UV matrix:
                # Rotate cylinder normal parallel to z-axis [0,0,1]
                orig_norm = mesh_part.normal_rep
                new_norm = np.array([0, 0, 1])
                v_c = np.cross(orig_norm, new_norm)
                # log.debug(" -- mesh_part.normal_rep: %s, new_norm: %s, v_c: %s", orig_norm, new_norm, v_c)

                # Project the mesh onto to x-y plane [x,y,z] -> [x+x*z, y+y*z, 0]
                projected_vertices = mesh_vertices.copy()
                # Rotate the vertices
                # 1. First, check if rotation is required by checking the magnitude of the transformation vector
                mag = np.sum([v_c[0]*v_c[0], v_c[1]*v_c[1], v_c[2]*v_c[2]])
                if mag > 0.0:
                    v_d = np.dot(orig_norm, new_norm)
                    projected_vertices = align_normals(projected_vertices, orig_norm, new_norm)
                    input_vertices = projected_vertices.copy()
                else:
                    input_vertices = mesh_vertices

                # Project the vertices onto the X-Y plane
                projected_vertices[:, 0] += input_vertices[:, 0] * input_vertices[:, 2]
                projected_vertices[:, 1] += input_vertices[:, 1] * input_vertices[:, 2]
                projected_vertices[:, 2] = 0  # Set z-coordinate to zero (projection onto x-y plane)
                projected_vertices_2d = projected_vertices[:,:2]

                mesh_uv = Mesh(vertices=projected_vertices, faces=mesh_faces)
                # Retrieve the vertices and the boundary loops of the projected cylinder
                mesh_part.uv = mesh_uv.get_vertices()
                mesh_part.boundary = get_boundary_loop_nodes(mesh_uv)

        else:
            # The 3D mesh is already planar, but the normals must be aligned to the z-axis
            log.debug(" - 3D mesh is already planar")

            # DEBUG
            # visualize_vertex_connections(mesh_vertices, 800, 'images/planar_projected1.png')

            # Rotate the planar mesh in the xy plane
            mean_norm = np.mean(face_normals, axis=0)
            new_norm = np.array([0, 0, 1])
            v_c = np.cross(mean_norm, new_norm)

            # log.debug(" - Orientation: mean_norm: %s, new_norm: %s, v_c: %s", mean_norm, new_norm, v_c)

            # Check if the normals are already aligned to the Z-axis
            if np.linalg.norm(v_c) > 1e-8:
                # Apply the rotation matrix to the vertices
                rotated_vertices = align_normals(mesh_vertices, mean_norm, new_norm)

                # Assign the rotated vertices to the UV attribute of the mesh
                mesh_part.uv = rotated_vertices[:, :2]
            else:
                # Assign the original vertex coordinates to the UV attribute of the mesh
                mesh_part.uv = mesh_vertices[:, :2]

            log.debug(" - mesh_part.uv shape: %s", mesh_part.uv.shape)

            mesh_part.boundary = get_boundary_loop_nodes(mesh_part)

    return coil_parts

# TODO: Consider moving these functions to other files
def align_normals(vertices, original_normal, desired_normal):
    """
    Aligns the normals of vertices with a desired normal vector.

    Args:
        vertices (ndarray): An Nx3 array of vertices [x, y, z].
        original_normal (ndarray): A 3D vector representing the original normal.
        desired_normal (ndarray): A 3D vector representing the desired normal.

    Returns:
        ndarray: Transformed vertices with aligned normals.
    """
    # Ensure the input vectors are normalized
    original_normal = original_normal / np.linalg.norm(original_normal)
    desired_normal = desired_normal / np.linalg.norm(desired_normal)

    v_c = np.cross(original_normal, desired_normal)
    v_d = np.dot(original_normal, desired_normal)

    # Construct the rotation matrix
    mat_v = np.array([[0, -v_c[2], v_c[1]],
                    [v_c[2], 0, -v_c[0]],
                    [-v_c[1], v_c[0], 0]])
    rot_mat = np.eye(3) + mat_v + np.matmul(mat_v, mat_v) * (1 / (1 + v_d))

    # Apply the rotation matrix to the vertices
    transformed_vertices = np.matmul(vertices, rot_mat.T)

    return transformed_vertices


def get_boundary_loop_nodes(mesh_part: Mesh):
    """
    Compute the indices of the vertices on the boundaries of the mesh.

    NOTE: This implementation only works for planar meshes.
    """
    # Compute the boundary edges of the mesh
    boundary_edges = mesh_part.boundary_edges()
    # log.debug(" - boundary_edges: %s -> %s", np.shape(boundary_edges), boundary_edges)

    # DEBUG
    # visualize_vertex_connections(mesh_part.uv, 800, 'images/boundary_edges1.png', boundary_edges)

    # Initialize variables
    boundary_loops = []
    visited = set()

    # Iterate through the boundary edges
    for facet in boundary_edges:
        for edge in facet:
            # log.debug(" - Edge: %s", edge)
            # Check if the edge has already been visited
            if edge[0] in visited or edge[1] in visited:
                continue

            # Start a new boundary loop
            boundary_loop = [edge[0]]
            current_vertex = edge[1]

            # Traverse the boundary loop
            while current_vertex != edge[0]:
                # Add the current vertex to the boundary loop
                boundary_loop.append(current_vertex)
                visited.add(current_vertex)

                # Find the next boundary edge connected to the current vertex
                next_edge = None
                for e in facet:
                    if e[0] == current_vertex and e[1] not in visited:
                        next_edge = e
                        # log.debug(" - next_edge: %s", next_edge)
                        break

                if next_edge is None:
                    # log.debug(" - No edge, trying again")
                    break
                # Update the current vertex
                current_vertex = next_edge[1]

            # Add the completed boundary loop to the list
            boundary_loops.append(boundary_loop)

    # DEBUG
    # total_elements = sum(len(sublist) for sublist in boundary_loops)
    # log.debug(" - boundary_loops shape: len(%s) -> %s", total_elements, boundary_loops)
    # return boundary_loops

    # Boundaries appear to be split into two windings, sharing the first element. Merge linked boundaries:
    reduced_loops = []
    boundary_part = boundary_loops[0]
    for index in range(1, len(boundary_loops)):
        next_part = boundary_loops[index]
        # Merge the loops if they share the same start vertex.
        if boundary_part[0] == next_part[0]:
            # Drop the first element and reverse the rest
            next_part = next_part[1:]
            next_part.reverse()
            boundary_part += next_part
        else:
            reduced_loops.append(boundary_part)
            boundary_part = next_part

    reduced_loops.append(boundary_part)
    # DEBUG
    # total_elements = sum(len(sublist) for sublist in reduced_loops)
    # log.debug(" - new_loops shape: len(%s) -> %s", total_elements, reduced_loops)

    return reduced_loops
