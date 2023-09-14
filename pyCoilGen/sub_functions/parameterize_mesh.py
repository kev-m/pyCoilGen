# System imports
from typing import List
import numpy as np

# Logging
import logging

# Local imports
from .data_structures import Mesh
from .mesh_parameterization_iterative import mesh_parameterization_iterative
from .calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector
from .constants import *

log = logging.getLogger(__name__)


def parameterize_mesh(coil_parts: List[Mesh], input_args) -> List[Mesh]:
    """
    Create the parameterized 2D mesh.

    Initialises the following properties of a CoilPart:
        - v,n (np.ndarray)  : vertices and vertex normals (m,3), (m,3)
        - f,fn (np.ndarray) : faces and face normals (n,2), (n,3)
        - uv (np.ndarray)   : 2D project of mesh (m,2)
        - boundary (int)    : list of lists boundary vertex indices (n, variable)

    Updates the following properties of a CoilPart:
        - None

    Depends on the following input_args:
        - surface_is_cylinder_flag
        - circular_diameter_factor

    Initialises the following properties of the CoilParts:
        - None

    Args:
        coil_parts (List[Mesh]): Coil parts list with attributes 'coil_mesh'.
        input_args (object): Input object with attributes 'surface_is_cylinder_flag' and 'circular_diameter_factor_cylinder_parameterization'.

    Returns:
        object: Updated coil parts object with parameterized mesh.
    """

    # The non-cylindrical parameterization is taken from "matlabmesh @ Ryan Schmidt  rms@dgp.toronto.edu"
    # based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes"

    surface_is_cylinder = input_args.surface_is_cylinder_flag
    circular_factor = input_args.circular_diameter_factor

    for part_ind in range(len(coil_parts)):
        mesh_part = coil_parts[part_ind].coil_mesh
        mesh_vertices = mesh_part.get_vertices()
        mesh_faces = mesh_part.get_faces()

        # DEBUG
        if input_args.debug > DEBUG_BASIC:
            log.debug(" - processing %d, vertices shape: %s", part_ind, mesh_part.get_vertices().shape)

        # Compute face and vertex normals
        face_normals = mesh_part.face_normals()
        vertex_normals = mesh_part.vertex_normals()

        max_face_normal = [np.std(face_normals[:, 0]),
                           np.std(face_normals[:, 1]),
                           np.std(face_normals[:, 2])]
        max_face_normal_std = np.max(max_face_normal)

        # DEBUG
        if input_args.debug > DEBUG_BASIC:
            log.debug(" - max_face_normal: %s, max_face_normal_std: %s", max_face_normal, max_face_normal_std)

        mesh_part.v = mesh_vertices
        mesh_part.f = mesh_faces
        mesh_part.fn = face_normals
        mesh_part.n = vertex_normals

        # Check if vertex coordinates are rather constant in one of the three dimensions
        if not (max_face_normal_std < 1e-6):
            # Go for the parameterization; distinguish between cylinder and non-cylinder
            if not surface_is_cylinder:
                # Create a 2D dataset for fit
                mesh_part = mesh_parameterization_iterative(mesh_part)
            else:
                # Planarization of cylinder

                # Create 2D mesh for UV matrix:
                # Rotate cylinder normal parallel to z-axis [0,0,1]
                orig_norm = mesh_part.normal_rep
                new_norm = np.array([0, 0, 1])
                v_c = np.cross(orig_norm, new_norm)
                # log.debug(" -- mesh_part.normal_rep: %s, new_norm: %s, v_c: %s", orig_norm, new_norm, v_c)

                # Find the boundaries by projecting the cylinder onto the X-Y plane:
                # Project the mesh onto to x-y plane
                projected_vertices = mesh_vertices.copy()
                # Rotate the vertices
                # 1. First, check if rotation is required by checking the magnitude of the transformation vector
                mag = np.sum([v_c[0]*v_c[0], v_c[1]*v_c[1], v_c[2]*v_c[2]])
                if mag > 0.0:
                    v_d = np.dot(orig_norm, new_norm)
                    projected_vertices = align_normals(projected_vertices, orig_norm, new_norm)
                    input_vertices = projected_vertices.copy()
                else:
                    input_vertices = mesh_vertices.copy()

                boundary_loop_nodes = mesh_part.boundary_indices()

                # MATLAB
                opening_mean = np.mean(mesh_vertices[boundary_loop_nodes[0], :], axis=0)  # -0.0141, -0.0141, -0.75
                overall_mean = np.mean(mesh_vertices, axis=0)

                old_orientation_vector = (opening_mean - overall_mean) / \
                    np.linalg.norm(opening_mean - overall_mean, axis=0)

                z_vec = np.array([0, 0, 1])
                sina = np.linalg.norm(np.cross(old_orientation_vector, z_vec)) / \
                    (np.linalg.norm(old_orientation_vector) * np.linalg.norm(z_vec))
                cosa = np.linalg.norm(np.dot(old_orientation_vector, z_vec)) / \
                    (np.linalg.norm(old_orientation_vector) * np.linalg.norm(z_vec))
                angle = np.arctan2(sina, cosa)
                cross_product = np.cross(old_orientation_vector, np.array([0, 0, 1]))
                rotation_vector = cross_product / np.linalg.norm(cross_product)

                rot_mat = calc_3d_rotation_matrix_by_vector(rotation_vector, angle)
                rotated_vertices = np.dot(input_vertices, rot_mat)

                # Calculate the UV matrix: MATLAB CoilGen method
                point_coords = rotated_vertices.T  # input_vertices.T
                min_z_cylinder = np.min(point_coords[2])  # Minimum of the Z co-ordinates # -0.7631
                point_coords[2] = point_coords[2] + min_z_cylinder  # Shift the Z-co-ordinates
                phi_coord = np.arctan2(point_coords[1], point_coords[0])
                r_coord = np.sqrt(point_coords[0]**2 + point_coords[1]**2)
                u_coord = (point_coords[2] - np.mean(r_coord) * circular_factor) * np.sin(phi_coord)
                v_coord = (point_coords[2] - np.mean(r_coord) * circular_factor) * np.cos(phi_coord)

                mesh_part.uv = np.vstack((u_coord, v_coord)).T
                mesh_part.boundary = boundary_loop_nodes
        else:
            # The 3D mesh is already planar, but the normals must be aligned to the z-axis
            # DEBUG
            if input_args.debug >= DEBUG_BASIC:
                log.debug(" - 3D mesh is already planar")

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

            # DEBUG
            if input_args.debug > DEBUG_BASIC:
                log.debug(" - mesh_part.uv shape: %s", mesh_part.uv.shape)

            mesh_part.boundary = mesh_part.boundary_indices()  # get_boundary_loop_nodes(mesh_part)

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
