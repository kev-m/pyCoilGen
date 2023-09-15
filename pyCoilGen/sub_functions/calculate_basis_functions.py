import numpy as np
from typing import List

# Logging
import logging

# Local imports
from .data_structures import CoilSolution, BasisElement, CoilPart
from .constants import get_level, DEBUG_VERBOSE

log = logging.getLogger(__name__)


def calculate_basis_functions(coil_parts: List[CoilPart]) -> List[CoilPart]:
    """
    Calculate the basis functions for the coil mesh.

    Initialises the following properties of a CoilPart:
        - basis_elements
        - is_real_triangle_mat
        - triangle_corner_coord_mat
        - current_mat
        - area_mat
        - face_normal_mat
        - current_density_mat

    Depends on the following properties of the CoilParts:
        - coil_mesh

    Depends on the following input_args:
        - None

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): List of CoilPart objects.

    Returns:
        list: Updated coil_parts with basis function information.
    """

    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        part_mesh = coil_part.coil_mesh
        part_vertices = part_mesh.get_vertices()  # Get the vertices for the coil part
        part_faces = part_mesh.get_faces()

        num_vertices = part_vertices.shape[0]
        num_faces = part_faces.shape[0]
        current_density_mat = np.zeros((num_vertices, num_faces, 3))

        # Create the container for the basis function
        coil_part.basis_elements = [BasisElement() for _ in range(num_vertices)]
        num_triangles_per_node = [len(coil_part.one_ring_list[vertex_index])
                                  for vertex_index in range(num_vertices)]

        max_triangle_count_per_node = 0

        for vertex_index in range(num_vertices):
            node_point = part_vertices[vertex_index]
            node_basis_element = coil_part.basis_elements[vertex_index]
            # Assign the triangle indices of the mesh faces
            node_basis_element.triangles = coil_part.node_triangles[vertex_index]
            # Assign stream function potential to this basis element
            node_basis_element.stream_function_potential = 0

            # Create basis structures
            node_triangles = num_triangles_per_node[vertex_index]
            node_basis_element.area = np.zeros((node_triangles))
            node_basis_element.face_normal = np.zeros((node_triangles, 3))
            node_basis_element.triangle_points_ABC = np.zeros((node_triangles, 3, 3))
            node_basis_element.current = np.zeros((node_triangles, 3))

            max_triangle_count_per_node = max(max_triangle_count_per_node, node_triangles)

            for tri_ind in range(node_triangles):
                point_b = part_vertices[coil_part.one_ring_list[vertex_index][tri_ind][0]]
                point_c = part_vertices[coil_part.one_ring_list[vertex_index][tri_ind][1]]
                node_basis_element.one_ring = coil_part.one_ring_list[vertex_index]
                # Calculate the area of the triangle
                node_basis_element.area[tri_ind] = np.linalg.norm(
                    np.cross(point_c - node_point, point_b - node_point)) / 2
                # Calculate the face normal of the triangle
                node_basis_element.face_normal[tri_ind] = np.cross(
                    point_c - node_point, point_c - point_b) / np.linalg.norm(np.cross(point_c - node_point, point_c - point_b))
                # Assign corner points ABC of the triangle
                node_basis_element.triangle_points_ABC[tri_ind] = np.asarray(
                    [node_point, point_b, point_c]).T  # Transposed to match MATLAB shape
                # Calculate the tangential current density of the triangle
                node_basis_element.current[tri_ind] = (point_c - point_b) / (2 * node_basis_element.area[tri_ind])
                # Calculate the current density matrix
                x = node_basis_element.triangles[tri_ind]
                current_density_mat[vertex_index, x] = node_basis_element.current[tri_ind]  # ??.T

                # DEBUG
                if get_level() > DEBUG_VERBOSE:
                    log.debug(" -- node_basis_element.current shape: %s", node_basis_element.current.shape)  # n x 3
                    log.debug(" -- node_basis_element.area shape: %s", node_basis_element.area.shape)  # n x 1
                    log.debug(" -- ")

        # Create outputs in matrix form
        is_real_triangle_mat = np.zeros((num_vertices, max_triangle_count_per_node), dtype=int)
        triangle_corner_coord_mat = np.zeros((num_vertices, max_triangle_count_per_node, 3, 3))
        face_normal_mat = np.zeros((num_vertices, max_triangle_count_per_node, 3))
        current_mat = np.zeros((num_vertices, max_triangle_count_per_node, 3))
        area_mat = np.zeros((num_vertices, max_triangle_count_per_node))

        for vertex_index in range(num_vertices):
            node_basis_element = coil_part.basis_elements[vertex_index]
            node_triangles = num_triangles_per_node[vertex_index]

            is_real_triangle_mat[vertex_index, :node_triangles] = True
            triangle_corner_coord_mat[vertex_index, is_real_triangle_mat[vertex_index]
                                      == 1] = node_basis_element.triangle_points_ABC

            current_mat[vertex_index, is_real_triangle_mat[vertex_index] == 1] = node_basis_element.current
            area_mat[vertex_index, is_real_triangle_mat[vertex_index] == 1] = node_basis_element.area

            face_normal_mat[vertex_index, is_real_triangle_mat[vertex_index] == 1
                            ] = node_basis_element.face_normal

        coil_part.is_real_triangle_mat = is_real_triangle_mat
        coil_part.triangle_corner_coord_mat = triangle_corner_coord_mat
        coil_part.current_mat = current_mat
        coil_part.area_mat = area_mat
        coil_part.face_normal_mat = face_normal_mat
        coil_part.current_density_mat = current_density_mat

    return coil_parts
