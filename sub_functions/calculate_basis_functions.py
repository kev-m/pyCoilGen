import numpy as np
from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilSolution, BasisElement, CoilPart
from sub_functions.constants import DEBUG_VERBOSE

log = logging.getLogger(__name__)


def calculate_basis_functions(coil_solution: CoilSolution, coil_parts: List[CoilPart], input):
    """
    Calculate the basis functions for the coil mesh.

    Args:
        coil_solution (CoilSolution): Coil solution object.
        coil_parts (list): List of CoilPart objects.
        input (object): Input object containing evaluation parameters.

    Returns:
        list: Updated coil_parts with basis function information.
    """
    optimisation = coil_solution.optimisation  # Retrieve the solution optimization parameters

    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        part_mesh = coil_part.coil_mesh
        part_vertices = part_mesh.get_vertices()  # Get the vertices for the coil part
        part_faces = part_mesh.get_faces()

        if not optimisation.use_preoptimization_temp:

            num_vertices = part_vertices.shape[0]
            num_faces = part_faces.shape[0]
            current_density_mat = np.zeros((num_vertices, num_faces, 3))

            # Create the container for the basis function
            coil_part.basis_elements = [BasisElement() for _ in range(num_vertices)]
            num_triangles_per_node = [len(coil_part.one_ring_list[vertex_index])
                                      for vertex_index in range(num_vertices)]

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

                for tri_ind in range(node_triangles):
                    point_b = part_vertices[coil_part.one_ring_list[vertex_index][tri_ind][0]]
                    point_c = part_vertices[coil_part.one_ring_list[vertex_index][tri_ind][1]]
                    node_basis_element.one_ring = coil_part.one_ring_list[vertex_index].T
                    # Calculate the area of the triangle
                    node_basis_element.area[tri_ind] = np.linalg.norm(
                        np.cross(point_c - node_point, point_b - node_point)) / 2
                    # Calculate the face normal of the triangle
                    node_basis_element.face_normal[tri_ind] = np.cross(
                        point_c - node_point, point_c - point_b) / np.linalg.norm(np.cross(point_c - node_point, point_c - point_b))
                    # Assign corner points ABC of the triangle
                    node_basis_element.triangle_points_ABC[tri_ind] = np.array(
                        [node_point, point_b, point_c]).T
                    # Calculate the tangential current density of the triangle
                    node_basis_element.current[tri_ind] = (point_c - point_b) / (2 * node_basis_element.area[tri_ind])
                    # Calculate the current density matrix
                    x = node_basis_element.triangles[tri_ind]
                    current_density_mat[vertex_index, x] = node_basis_element.current[tri_ind].T

                    # DEBUG
                    if input.debug > DEBUG_VERBOSE:
                        log.debug(" -- node_basis_element.current shape: %s", node_basis_element.current.shape)  # n x 3
                        log.debug(" -- node_basis_element.area shape: %s", node_basis_element.area.shape)  # n x 1
                        log.debug(" -- ")

            # Create outputs in matrix form
            highst_triangle_count_per_node = max(
                len(node_basis_element.area) for node_ind in range(num_vertices))
            is_real_triangle_mat = np.zeros((num_vertices, highst_triangle_count_per_node), dtype=bool)
            triangle_corner_coord_mat = np.zeros((num_vertices, highst_triangle_count_per_node, 3, 3))
            face_normal_mat = np.zeros((num_vertices, highst_triangle_count_per_node, 3))
            current_mat = np.zeros((num_vertices, highst_triangle_count_per_node, 3))
            area_mat = np.zeros((num_vertices, highst_triangle_count_per_node))

            for vertex_index in range(num_vertices):
                is_real_triangle_mat[vertex_index, :len(node_basis_element.area)] = True
                triangle_corner_coord_mat[vertex_index, is_real_triangle_mat[vertex_index]
                                          ] = node_basis_element.triangle_points_ABC

                current_mat[vertex_index, is_real_triangle_mat[vertex_index]] = node_basis_element.current
                area_mat[vertex_index, is_real_triangle_mat[vertex_index]] = node_basis_element.area

                face_normal_mat[vertex_index, is_real_triangle_mat[vertex_index]
                                ] = node_basis_element.face_normal

            coil_part.is_real_triangle_mat = is_real_triangle_mat
            coil_part.triangle_corner_coord_mat = triangle_corner_coord_mat
            coil_part.current_mat = current_mat
            coil_part.area_mat = area_mat
            coil_part.face_normal_mat = face_normal_mat
            coil_part.current_density_mat = current_density_mat

        else:
            raise Exception("Optimisation is not implemented!")
            coil_part.is_real_triangle_mat = input.temp.coil_parts[part_ind].is_real_triangle_mat
            coil_part.triangle_corner_coord_mat = input.temp.coil_parts[part_ind].triangle_corner_coord_mat
            coil_part.current_mat = input.temp.coil_parts[part_ind].current_mat
            coil_part.area_mat = input.temp.coil_parts[part_ind].area_mat
            coil_part.face_normal_mat = input.temp.coil_parts[part_ind].face_normal_mat
            coil_part.current_density_mat = input.temp.coil_parts[part_ind].current_density_mat

    return coil_parts


"""
def calculate_basis_functions(coil_solution : CoilSolution, coil_parts: List[CoilPart], input):
    optimisation = coil_solution.optimisation # Retrieve the solution optmisation parameters
    # Initialize the outputs
    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        part_mesh = coil_part.coil_mesh
        part_vertices = part_mesh.get_vertices() # Get the vertices for the coil part.
        coil_part.is_real_triangle_mat = np.zeros((0, 0), dtype=bool)
        coil_part.triangle_corner_coord_mat = np.zeros((0, 0, 3, 3))
        coil_part.current_mat = np.zeros((0, 0, 3))
        coil_part.area_mat = np.zeros((0, 0))
        coil_part.face_normal_mat = np.zeros((0, 0, 3))
        coil_part.basis_elements = []
        coil_part.current_density_mat = np.zeros((0, 0, 3))

        if not optimisation.use_preoptimization_temp:
            num_nodes = part_vertices.shape[1]
            current_density_mat = np.zeros((num_nodes, part_mesh.faces.shape[1], 3))

            # Create the container for the basis function
            coil_part.basis_elements = [BasisElement() for _ in range(num_nodes)]
            num_triangles_per_node = [len(coil_part.one_ring_list[i]) for i in range(num_nodes)]

            for node_ind in range(num_nodes):
                node_point = part_vertices[:, node_ind]
                node_basis_element.triangles = coil_part.node_triangles[node_ind]
                node_basis_element.stream_function_potential = 0

                for tri_ind in range(num_triangles_per_node[node_ind]):
                    point_b = part_vertices[:,coil_part.one_ring_list[node_ind][0, tri_ind]]
                    point_c = part_vertices[:,coil_part.one_ring_list[node_ind][1, tri_ind]]

                    node_basis_element.one_ring = coil_part.one_ring_list[node_ind]
                    node_basis_element.area[tri_ind] = np.linalg.norm(
                        np.cross(point_c - node_point, point_b - node_point)
                    ) / 2

                    node_basis_element.face_normal[tri_ind] = np.cross(
                        point_c - node_point, point_c - point_b
                    ) / np.linalg.norm(np.cross(point_c - node_point, point_c - point_b))

                    node_basis_element.triangle_points_ABC[tri_ind] = np.array(
                        [node_point, point_b, point_c]
                    )

                    node_basis_element.current[tri_ind] = (
                        point_c - point_b
                    ) / (2 * node_basis_element.area[tri_ind])

                    # Calculate the current density matrix
                    current_density_mat[node_ind, node_basis_element.triangles[tri_ind]] = \
                        node_basis_element.current[tri_ind]

            # Create outputs in matrix form
            highst_triangle_count_per_node = max(
                [len(coil_part.basis_elements[i].area) for i in range(num_nodes)])
            is_real_triangle_mat = np.zeros((num_nodes, highst_triangle_count_per_node), dtype=bool)
            triangle_corner_coord_mat = np.zeros((num_nodes, highst_triangle_count_per_node, 3, 3))
            face_normal_mat = np.zeros((num_nodes, highst_triangle_count_per_node, 3))
            current_mat = np.zeros((num_nodes, highst_triangle_count_per_node, 3))
            area_mat = np.zeros((num_nodes, highst_triangle_count_per_node))

            for node_ind in range(num_nodes):
                is_real_triangle_mat[node_ind, :len(
                    node_basis_element.area)] = True
                triangle_corner_coord_mat[node_ind, is_real_triangle_mat[node_ind]] = \
                    node_basis_element.triangle_points_ABC
                current_mat[node_ind, is_real_triangle_mat[node_ind]] = \
                    node_basis_element.current
                area_mat[node_ind, is_real_triangle_mat[node_ind]] = \
                    node_basis_element.area
                face_normal_mat[node_ind, is_real_triangle_mat[node_ind]] = \
                    node_basis_element.face_normal

            coil_part.is_real_triangle_mat = is_real_triangle_mat
            coil_part.triangle_corner_coord_mat = triangle_corner_coord_mat
            coil_part.current_mat = current_mat
            coil_part.area_mat = area_mat
            coil_part.face_normal_mat = face_normal_mat
            coil_part.current_density_mat = current_density_mat

        else:
            raise Exception("Optimisation is not implemented!")
            # Use pre-optimized data from input object
            coil_part.is_real_triangle_mat = optimisation.coil_part.is_real_triangle_mat
            coil_part.triangle_corner_coord_mat = optimisation.coil_part.triangle_corner_coord_mat
            coil_part.current_mat = optimisation.coil_part.current_mat
            coil_part.area_mat = optimisation.coil_part.area_mat
            coil_part.face_normal_mat = optimisation.coil_part.face_normal_mat
            coil_part.current_density_mat = optimisation.coil_part.current_density_mat
"""
