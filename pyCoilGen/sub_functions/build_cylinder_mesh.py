import numpy as np

# Logging
import logging

# Local imports
from .calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector
from .data_structures import DataStructure

log = logging.getLogger(__name__)


def build_cylinder_mesh(
        cylinder_height,
        cylinder_radius,
        num_circular_divisions,
        num_longitudinal_divisions,
        rotation_vector_x,
        rotation_vector_y,
        rotation_vector_z,
        rotation_angle
):
    """
    Create a cylindrical regular mesh in any orientation.

    Args:
        cylinder_height (float): Height of the cylinder.
        cylinder_radius (float): Radius of the cylinder.
        num_circular_divisions (int): Number of circular divisions.
        num_longitudinal_divisions (int): Number of longitudinal divisions.
        rotation_vector_x (float): X-component of the rotation vector.
        rotation_vector_y (float): Y-component of the rotation vector.
        rotation_vector_z (float): Z-component of the rotation vector.
        rotation_angle (float): Rotation angle.

    Returns:
        mesh: DataStructure with 'faces' and 'vertices' arrays of the cylindrical mesh.
    """
    # Calculate x, y, z positions of the vertices
    x_positions = np.sin(np.linspace(0, 2 * np.pi, num_circular_divisions + 1)) * cylinder_radius
    y_positions = np.cos(np.linspace(0, 2 * np.pi, num_circular_divisions + 1)) * cylinder_radius
    x_positions = x_positions[:-1]  # Remove repetition at the end
    y_positions = y_positions[:-1]  # Remove repetition at the end
    z_positions = np.linspace(-cylinder_height / 2, cylinder_height / 2,
                              num_longitudinal_divisions+2)
    z_positions = z_positions[:-1]  # Remove repetition at the end

    # Create the mesh vertices
    vertices_x = np.tile(x_positions, num_longitudinal_divisions+1)
    vertices_y = np.tile(y_positions, num_longitudinal_divisions+1)
    vertices_z = np.repeat(z_positions, len(x_positions))
    vertices = np.vstack((vertices_x, vertices_y, vertices_z))

    # Set the vertices in the center
    vertices = vertices - np.mean(vertices, axis=1, keepdims=True)

    # Create the faces for the cylinder mesh
    tri_1_vert_inds_1 = np.arange(num_circular_divisions * num_longitudinal_divisions)
    tri_1_vert_inds_2 = tri_1_vert_inds_1 + 1
    tri_1_vert_inds_2[tri_1_vert_inds_2 % num_circular_divisions == 0] -= num_circular_divisions
    tri_1_vert_inds_3 = tri_1_vert_inds_2 + num_circular_divisions

    tri_2_vert_inds_1 = tri_1_vert_inds_1
    tri_2_vert_inds_2 = tri_1_vert_inds_3
    tri_2_vert_inds_3 = np.arange(num_circular_divisions * num_longitudinal_divisions) + num_circular_divisions

    faces_1 = np.column_stack((tri_1_vert_inds_2, tri_1_vert_inds_1, tri_1_vert_inds_3))
    faces_2 = np.column_stack((tri_2_vert_inds_2, tri_2_vert_inds_1, tri_2_vert_inds_3))
    faces = np.vstack((faces_1, faces_2))  # Subtract 1 due to Matlab index offset starting at 1

    # Rotate the cylinder in the desired orientation
    rot_vec = np.array([rotation_vector_x, rotation_vector_y, rotation_vector_z])
    rot_mat = calc_3d_rotation_matrix_by_vector(rot_vec, rotation_angle)
    vertices = np.dot(vertices.T, rot_mat)

    # Calculate representative normal
    normal = np.array([0.0, 0.0, 1.0])
    cylinder_mesh = DataStructure(vertices=vertices, faces=faces, normal=normal)

    return cylinder_mesh
