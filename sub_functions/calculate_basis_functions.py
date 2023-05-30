import numpy as np
from typing import List

from data_structures import BasisElement, CoilPart


def calculate_basis_functions(coil_parts: List[CoilPart], input):
    # Initialize the outputs
    for part_ind in range(len(coil_parts)):
        coil_parts[part_ind].is_real_triangle_mat = np.zeros(
            (0, 0), dtype=bool)
        coil_parts[part_ind].triangle_corner_coord_mat = np.zeros((0, 0, 3, 3))
        coil_parts[part_ind].current_mat = np.zeros((0, 0, 3))
        coil_parts[part_ind].area_mat = np.zeros((0, 0))
        coil_parts[part_ind].face_normal_mat = np.zeros((0, 0, 3))
        coil_parts[part_ind].basis_elements = []
        coil_parts[part_ind].current_density_mat = np.zeros((0, 0, 3))

    for part_ind in range(len(coil_parts)):
        if not input.temp_evalution.use_preoptimization_temp:
            num_nodes = coil_parts[part_ind].coil_mesh.vertices.shape[1]
            current_density_mat = np.zeros(
                (num_nodes, coil_parts[part_ind].coil_mesh.faces.shape[1], 3))

            # Create the container for the basis function
            coil_parts[part_ind].basis_elements = [BasisElement()
                                                   for _ in range(num_nodes)]
            num_triangles_per_node = [
                len(coil_parts[part_ind].one_ring_list[i]) for i in range(num_nodes)]

            for node_ind in range(num_nodes):
                node_point = coil_parts[part_ind].coil_mesh.vertices[:, node_ind]
                coil_parts[part_ind].basis_elements[node_ind].triangles = coil_parts[part_ind].node_triangles[node_ind]
                coil_parts[part_ind].basis_elements[node_ind].stream_function_potential = 0

                for tri_ind in range(num_triangles_per_node[node_ind]):
                    point_b = coil_parts[part_ind].coil_mesh.vertices[:,
                                                                      coil_parts[part_ind].one_ring_list[node_ind][0, tri_ind]]
                    point_c = coil_parts[part_ind].coil_mesh.vertices[:,
                                                                      coil_parts[part_ind].one_ring_list[node_ind][1, tri_ind]]

                    coil_parts[part_ind].basis_elements[node_ind].one_ring = coil_parts[part_ind].one_ring_list[node_ind]
                    coil_parts[part_ind].basis_elements[node_ind].area[tri_ind] = np.linalg.norm(
                        np.cross(point_c - node_point, point_b - node_point)
                    ) / 2

                    coil_parts[part_ind].basis_elements[node_ind].face_normal[tri_ind] = np.cross(
                        point_c - node_point, point_c - point_b
                    ) / np.linalg.norm(np.cross(point_c - node_point, point_c - point_b))

                    coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC[tri_ind] = np.array(
                        [node_point, point_b, point_c]
                    )

                    coil_parts[part_ind].basis_elements[node_ind].current[tri_ind] = (
                        point_c - point_b
                    ) / (2 * coil_parts[part_ind].basis_elements[node_ind].area[tri_ind])

                    # Calculate the current density matrix
                    current_density_mat[node_ind, coil_parts[part_ind].basis_elements[node_ind].triangles[tri_ind]] = \
                        coil_parts[part_ind].basis_elements[node_ind].current[tri_ind]

            # Create outputs in matrix form
            highst_triangle_count_per_node = max(
                [len(coil_parts[part_ind].basis_elements[i].area) for i in range(num_nodes)])
            is_real_triangle_mat = np.zeros(
                (num_nodes, highst_triangle_count_per_node), dtype=bool)
            triangle_corner_coord_mat = np.zeros(
                (num_nodes, highst_triangle_count_per_node, 3, 3))
            face_normal_mat = np.zeros(
                (num_nodes, highst_triangle_count_per_node, 3))
            current_mat = np.zeros(
                (num_nodes, highst_triangle_count_per_node, 3))
            area_mat = np.zeros((num_nodes, highst_triangle_count_per_node))

            for node_ind in range(num_nodes):
                is_real_triangle_mat[node_ind, :len(
                    coil_parts[part_ind].basis_elements[node_ind].area)] = True
                triangle_corner_coord_mat[node_ind, is_real_triangle_mat[node_ind]] = \
                    coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC
                current_mat[node_ind, is_real_triangle_mat[node_ind]] = \
                    coil_parts[part_ind].basis_elements[node_ind].current
                area_mat[node_ind, is_real_triangle_mat[node_ind]] = \
                    coil_parts[part_ind].basis_elements[node_ind].area
                face_normal_mat[node_ind, is_real_triangle_mat[node_ind]] = \
                    coil_parts[part_ind].basis_elements[node_ind].face_normal

            coil_parts[part_ind].is_real_triangle_mat = is_real_triangle_mat
            coil_parts[part_ind].triangle_corner_coord_mat = triangle_corner_coord_mat
            coil_parts[part_ind].current_mat = current_mat
            coil_parts[part_ind].area_mat = area_mat
            coil_parts[part_ind].face_normal_mat = face_normal_mat
            coil_parts[part_ind].current_density_mat = current_density_mat

        else:
            coil_parts[part_ind].is_real_triangle_mat = input.temp.coil_parts[part_ind].is_real_triangle_mat
            coil_parts[part_ind].triangle_corner_coord_mat = input.temp.coil_parts[part_ind].triangle_corner_coord_mat
            coil_parts[part_ind].current_mat = input.temp.coil_parts[part_ind].current_mat
            coil_parts[part_ind].area_mat = input.temp.coil_parts[part_ind].area_mat
            coil_parts[part_ind].face_normal_mat = input.temp.coil_parts[part_ind].face_normal_mat
            coil_parts[part_ind].current_density_mat = input.temp.coil_parts[part_ind].current_density_mat
