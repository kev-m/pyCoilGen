import numpy as np

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector
from pyCoilGen.sub_functions.data_structures import DataStructure
from pyCoilGen.sub_functions.read_mesh import create_unique_noded_mesh
from pyCoilGen.helpers.common import int_or_float

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
                              num_longitudinal_divisions+1)

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


def create_cylinder_mesh(input_args):
    """Template function to create a cylinder mesh.

    Used when 'input_args.coil_mesh_file' is 'create cylinder mesh'.
    """
    log.debug("Creating cylinder mesh with '%s'", input_args.cylinder_mesh_parameter_list)
    mesh_data = build_cylinder_mesh(*input_args.cylinder_mesh_parameter_list)
    coil_mesh = create_unique_noded_mesh(mesh_data)
    return coil_mesh


__default_value__ = [0.8, 0.3, 20, 20, 1, 0, 0, 0]


def get_name():
    """
    Template function to retrieve the plugin builder name.

    Returns:
        builder_name (str): The builder name, given to 'coil_mesh'.
    """
    return 'create cylinder mesh'


def get_parameters() -> list:
    """
    Template function to retrieve the supported parameters and default values as strings.

    Returns:
        list of tuples of parameter name and default value: The additional parameters provided by this builder
    """
    return [('cylinder_mesh_parameter_list', str(__default_value__))]


def register_args(parser):
    """Template function to register arguments specific to planar mesh creation.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    # Add the parameters for the generation of the (default) cylindrical mesh
    parser.add_argument('--cylinder_mesh_parameter_list', nargs='+', type=int_or_float,
                        default=__default_value__,
                        help="Parameters for the generation of the (default) cylindrical mesh")
