import numpy as np
from scipy.spatial import Delaunay

def parameterize_mesh(coil_parts, input):
    """
    Create the parameterized 2D mesh.

    Args:
        coil_parts (object): Coil parts object with attributes 'coil_mesh'.
        input (object): Input object with attributes 'surface_is_cylinder_flag' and 'circular_diameter_factor_cylinder_parameterization'.

    Returns:
        object: Updated coil parts object with parameterized mesh.
    """

    # The non-cylindrical parameterization is taken from "matlabmesh @ Ryan
    # Schmidt  rms@dgp.toronto.edu" based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes"

    surface_is_cylinder = input.surface_is_cylinder_flag
    circular_factor = input.circular_diameter_factor_cylinder_parameterization

    for part_ind in range(len(coil_parts)):

        # Compute face normals
        face_normals = faceNormal(triangulation(coil_parts[part_ind].coil_mesh.faces.T, coil_parts[part_ind].coil_mesh.vertices.T))
        max_face_normal_std = np.max([np.std(face_normals[:, 0]), np.std(face_normals[:, 1]), np.std(face_normals[:, 2])])

        coil_parts[part_ind].coil_mesh.v = coil_parts[part_ind].coil_mesh.vertices.T
        coil_parts[part_ind].coil_mesh.fn = faceNormal(triangulation(coil_parts[part_ind].coil_mesh.faces.T, coil_parts[part_ind].coil_mesh.v))

        # Check if vertex coordinates are rather constant in one of the three dimensions
        if not (max_face_normal_std < 1e-6):
            # Go for the parameterization; distinguish between cylinder and non-cylinder

            if not surface_is_cylinder:
                coil_parts[part_ind].coil_mesh = mesh_parameterization_iterative(coil_parts[part_ind].coil_mesh)
                # Create a 2D dataset for fit
            else:
                # Planarization of cylinder
                boundary_edges = freeBoundary(triangulation(coil_parts[part_ind].coil_mesh.faces.T, coil_parts[part_ind].coil_mesh.vertices.T))

                # Build the boundary loops from the boundary edges
                is_new_node = np.hstack(([boundary_edges[:, 0].T], [0])) == np.hstack(([0], [boundary_edges[:, 1].T]))
                is_new_node[0] = True
                is_new_node[-1] = True
                is_new_node = ~is_new_node
                num_boundaries = np.sum(is_new_node) + 1
                boundary_start = np.hstack(([1], np.where(is_new_node)[0] + 1))
                boundary_end = np.hstack((np.where(is_new_node)[0], [boundary_edges.shape[0] - 1]))
                boundary_loop_nodes = []

                for boundary_ind in range(num_boundaries):
                    boundary_loop_nodes.append(np.hstack((boundary_edges[boundary_start[boundary_ind]:boundary_end[boundary_ind], 0], boundary_edges[boundary_start[boundary_ind], 0])))

                # Check if the cylinder is oriented along the z-axis
                # If so, make a rotated copy for the parameterization
                opening_mean = np.mean(coil_parts[part_ind].coil_mesh.vertices[:, boundary_loop_nodes[0]], axis=1)
                overall_mean = np.mean(coil_parts[part_ind].coil_mesh.vertices, axis=1)
                old_orientation_vector = (opening_mean - overall_mean) / np.linalg.norm(opening_mean - overall_mean)
                z_vec = np.array([0, 0, 1])
                sina = np.linalg.norm(np.cross(old_orientation_vector, z_vec)) / (np.linalg.norm(old_orientation_vector) * np.linalg.norm(z_vec))
                cosa = np.dot(old_orientation_vector, z_vec) / (np.linalg.norm(old_orientation_vector) * np.linalg.norm(z_vec))
                angle = np.arctan2(sina, cosa)
                rotation_vector = np.cross(old_orientation_vector, np.array([0, 0, 1])) / np.linalg.norm(np.cross(old_orientation_vector, np.array([0, 0, 1])))

                # Calculate 3D rotation matrix by vector and angle
                rot_mat = calc_3d_rotation_matrix_by_vector(rotation_vector, angle)
                rotated_vertices = np.dot(rot_mat, coil_parts[part_ind].coil_mesh.vertices)
                point_coords = rotated_vertices
                min_z_cylinder = np.min(point_coords[2, :])
                point_coords[2, :] = point_coords[2, :] + min_z_cylinder
                phi_coord = np.arctan2(point_coords[1, :], point_coords[0, :])
                r_coord = np.sqrt(point_coords[0, :]**2 + point_coords[1, :]**2)
                u_coord = (point_coords[2, :] - np.mean(r_coord) * circular_factor) * np.sin(phi_coord)
                v_coord = (point_coords[2, :] - np.mean(r_coord) * circular_factor) * np.cos(phi_coord)

                coil_parts[part_ind].coil_mesh.uv = np.vstack((u_coord, v_coord))
                coil_parts[part_ind].coil_mesh.n = vertexNormal(triangulation(coil_parts[part_ind].coil_mesh.faces.T, coil_parts[part_ind].coil_mesh.vertices.T)).T
                coil_parts[part_ind].coil_mesh.boundary = boundary_loop_nodes





    return coil_parts
