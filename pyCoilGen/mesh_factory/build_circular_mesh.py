import numpy as np
# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import DataStructure, Mesh
from pyCoilGen.sub_functions.read_mesh import create_unique_noded_mesh
from pyCoilGen.helpers.triangulation import Triangulate
from pyCoilGen.sub_functions.calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector
from pyCoilGen.helpers.common import int_or_float

log = logging.getLogger(__name__)


def build_circular_mesh(radius: float, num_radial_divisions: int,
                        rotation_vector_x: float, rotation_vector_y: float, rotation_vector_z: float,
                        rotation_angle: float,
                        center_position_x: float, center_position_y: float, center_position_z: float) -> DataStructure:
    """
    Create a mono-planar circular mesh in any orientation.

    Args:
        radius (float): Radius of the circular mesh.
        num_radial_divisions (int): Number of radial divisions.
        rotation_vector_x (float): X component of the rotation vector.
        rotation_vector_y (float): Y component of the rotation vector.
        rotation_vector_z (float): Z component of the rotation vector.
        rotation_angle (float): Rotation angle in radians.
        center_position_x (float): X component of the center position.
        center_position_y (float): Y component of the center position.
        center_position_z (float): Z component of the center position.

    Returns:
        DataStructure(vertices, faces): DataStructure containing the vertices and faces of the planar mesh.

    Raises:
        ValueError: If the provided radius is not positive.

    Example:
        >>> parameters = {
        ...     'radius' : 1.0,
        ...     'num_radial_divisions' : 10,
        ...     'rotation_vector_x' : 0.0,
        ...     'rotation_vector_y' : 0.0,
        ...     'rotation_vector_z' : 1.0,
        ...     'rotation_angle' : 0.0,
        ...     'center_position_x' : 0.0,
        ...     'center_position_y' : 0.0,
        ...     'center_position_z' : 0.0
        ... }
        >>> circular_mesh = build_circular_mesh(**parameters)
    """
    if radius <= 0:
        raise ValueError("Radius must be positive.")

    rotation_vector = np.array([rotation_vector_x, rotation_vector_y, rotation_vector_z])
    center_position = np.array([center_position_x, center_position_y, center_position_z])

    # Generate x and y positions
    x_positions = np.linspace(-radius, radius, num_radial_divisions+1)
    y_positions = np.linspace(-radius, radius, num_radial_divisions+1)

    # Create x and y grids
    x, y = np.meshgrid(x_positions, y_positions)

    # Define the vertex positions
    vertices = np.vstack((y.ravel(), x.ravel(), np.zeros_like(x.ravel())))

    # Define the mesh triangles
    tri_1_vert_inds_1 = (np.tile(np.arange(num_radial_divisions), num_radial_divisions) +
                         np.repeat(np.arange(num_radial_divisions), num_radial_divisions) * (num_radial_divisions + 1))
    tri_1_vert_inds_2 = tri_1_vert_inds_1 + 1
    tri_1_vert_inds_3 = tri_1_vert_inds_2 + num_radial_divisions + 1
    tri_2_vert_inds_1 = tri_1_vert_inds_1
    tri_2_vert_inds_2 = tri_1_vert_inds_3
    tri_2_vert_inds_3 = tri_1_vert_inds_3 - 1

    faces_1 = np.column_stack((tri_1_vert_inds_1, tri_1_vert_inds_3, tri_1_vert_inds_2))
    faces_2 = np.column_stack((tri_2_vert_inds_1, tri_2_vert_inds_3, tri_2_vert_inds_2))

    circular_mesh_faces = np.vstack((faces_1, faces_2))
    circular_mesh_vertices = vertices.T

    # Create circular mesh
    normal_rep = np.array([0.0, 0.0, 1.0])
    circular_mesh = DataStructure(faces=circular_mesh_faces, vertices=circular_mesh_vertices, normal=normal_rep)

    # Delete vertices outside the circular boundary
    vert_distances = np.linalg.norm(circular_mesh.vertices, axis=1)
    vertices_to_delete = np.where(vert_distances > radius * 0.99)[0][::-1]

    for delete_ind in vertices_to_delete:
        circular_mesh.vertices = np.delete(circular_mesh.vertices, delete_ind, axis=0)

        # Remove faces related to the deleted vertex and update the index list
        faces_to_delete = np.any(circular_mesh.faces == delete_ind, axis=1)
        circular_mesh.faces = circular_mesh.faces[~faces_to_delete]

        # Update face indices
        circular_mesh.faces[circular_mesh.faces > delete_ind] -= 1

    # Morph boundary verts for proper circular shape
    mesh = Mesh(vertices=circular_mesh.vertices, faces=circular_mesh.faces)
    boundary_verts = mesh.boundary_indices()
    for vert_ind in boundary_verts[0]:
        vert_radius = np.linalg.norm(circular_mesh.vertices[vert_ind, :2])
        circular_mesh.vertices[vert_ind, :] *= radius / vert_radius

    # Adjust mesh with desired translation and rotation
    rotation_matrix = calc_3d_rotation_matrix_by_vector(rotation_vector, rotation_angle)
    circular_mesh.vertices = np.dot(circular_mesh.vertices, rotation_matrix) + center_position

    return circular_mesh


def create_circular_mesh(input_args):
    """Template function to create a circular mesh.

    Used when 'input_args.coil_mesh_file' is 'create circular mesh'.
    """
    log.debug("Creating circular mesh with '%s'", input_args.circular_mesh_parameter_list)
    mesh_data = build_circular_mesh(*input_args.circular_mesh_parameter_list)
    coil_mesh = create_unique_noded_mesh(mesh_data)
    return coil_mesh


__default_value__ = [0.25, 20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]


def get_name():
    """
    Template function to retrieve the plugin builder name.

    Returns:
        builder_name (str): The builder name, given to 'coil_mesh'.
    """
    return 'create circular mesh'


def get_parameters() -> list:
    """
    Template function to retrieve the supported parameters and default values as strings.

    Returns:
        list of tuples of parameter name and default value: The additional parameters provided by this builder
    """
    return [('circular_mesh_parameter_list', str(__default_value__))]


def register_args(parser):
    """Template function to register arguments specific to planar mesh creation.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    # Add the parameters for the generation of the (default) planar mesh
    parser.add_argument('--circular_mesh_parameter_list', nargs='+', type=int_or_float,
                        default=__default_value__,
                        help="Parameters for the generation of the (default) planar mesh")


if __name__ == "__main__":
    radius = 0.5
    num_radial_divisions = 20
    rotation_vector_x = 0.0
    rotation_vector_y = 0.0
    rotation_vector_z = 1.0
    rotation_angle = 0.0
    center_position_x = 0.0
    center_position_y = 0.0
    center_position_z = 0.0

    circular_mesh = build_circular_mesh(radius, num_radial_divisions,
                                        rotation_vector_x, rotation_vector_y, rotation_vector_z,
                                        rotation_angle,
                                        center_position_x, center_position_y, center_position_z)

    mesh = Mesh(vertices=circular_mesh.vertices, faces=circular_mesh.faces)
    mesh.display()
