import numpy as np
from typing import List

# Logging
import logging

# Local imports
from .data_structures import CoilSolution, BasisElement, CoilPart

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

        # The triangles of all nodes, flattened: triangle k belongs to node `node_of_triangle[k]`, and is the
        # `position_in_node[k]`-th triangle of that node. The one ring of a node gives the other two corners (b, c).
        one_ring_list = coil_part.one_ring_list
        num_triangles_per_node = np.array([len(one_ring_list[vertex_index]) for vertex_index in range(num_vertices)],
                                          dtype=int)
        num_triangles = int(num_triangles_per_node.sum())
        node_starts = np.cumsum(num_triangles_per_node) - num_triangles_per_node
        node_of_triangle = np.repeat(np.arange(num_vertices), num_triangles_per_node)
        position_in_node = np.arange(num_triangles) - node_starts[node_of_triangle]
        if num_triangles:
            corner_indices = np.concatenate([np.reshape(one_ring_list[vertex_index], (-1, 2))
                                             for vertex_index in range(num_vertices)
                                             if num_triangles_per_node[vertex_index]]).astype(int)
            face_of_triangle = np.concatenate([np.asarray(coil_part.node_triangles[vertex_index])[
                :num_triangles_per_node[vertex_index]] for vertex_index in range(num_vertices)
                if num_triangles_per_node[vertex_index]]).astype(int)
        else:
            corner_indices = np.zeros((0, 2), dtype=int)
            face_of_triangle = np.zeros(0, dtype=int)

        # The geometry of every triangle
        point_a = part_vertices[node_of_triangle]
        point_b = part_vertices[corner_indices[:, 0]]
        point_c = part_vertices[corner_indices[:, 1]]
        area_all = np.linalg.norm(np.cross(point_c - point_a, point_b - point_a), axis=1) / 2
        normal_direction = np.cross(point_c - point_a, point_c - point_b)
        face_normal_all = normal_direction / np.linalg.norm(normal_direction, axis=1, keepdims=True)
        # [triangle, xyz, corner], the MATLAB shape of triangle_points_ABC
        triangle_points_all = np.stack((point_a, point_b, point_c), axis=2)
        current_all = (point_c - point_b) / (2 * area_all)[:, np.newaxis]

        current_density_mat = np.zeros((num_vertices, num_faces, 3))
        current_density_mat[node_of_triangle, face_of_triangle] = current_all

        # Create the container for the basis function
        coil_part.basis_elements = [BasisElement() for _ in range(num_vertices)]
        max_triangle_count_per_node = int(num_triangles_per_node.max()) if num_vertices else 0

        for vertex_index in range(num_vertices):
            node_basis_element = coil_part.basis_elements[vertex_index]
            first = node_starts[vertex_index]
            last = first + num_triangles_per_node[vertex_index]
            node_basis_element.triangles = coil_part.node_triangles[vertex_index]
            # Assign stream function potential to this basis element
            node_basis_element.stream_function_potential = 0
            node_basis_element.one_ring = one_ring_list[vertex_index]
            node_basis_element.area = area_all[first:last].copy()
            node_basis_element.face_normal = face_normal_all[first:last].copy()
            node_basis_element.triangle_points_ABC = triangle_points_all[first:last].copy()
            node_basis_element.current = current_all[first:last].copy()

        # Create outputs in matrix form, padded to the maximum number of triangles per node
        is_real_triangle_mat = np.zeros((num_vertices, max_triangle_count_per_node), dtype=int)
        triangle_corner_coord_mat = np.zeros((num_vertices, max_triangle_count_per_node, 3, 3))
        face_normal_mat = np.zeros((num_vertices, max_triangle_count_per_node, 3))
        current_mat = np.zeros((num_vertices, max_triangle_count_per_node, 3))
        area_mat = np.zeros((num_vertices, max_triangle_count_per_node))

        is_real_triangle_mat[node_of_triangle, position_in_node] = 1
        triangle_corner_coord_mat[node_of_triangle, position_in_node] = triangle_points_all
        current_mat[node_of_triangle, position_in_node] = current_all
        area_mat[node_of_triangle, position_in_node] = area_all
        face_normal_mat[node_of_triangle, position_in_node] = face_normal_all

        coil_part.is_real_triangle_mat = is_real_triangle_mat
        coil_part.triangle_corner_coord_mat = triangle_corner_coord_mat
        coil_part.current_mat = current_mat
        coil_part.area_mat = area_mat
        coil_part.face_normal_mat = face_normal_mat
        coil_part.current_density_mat = current_density_mat

    return coil_parts
