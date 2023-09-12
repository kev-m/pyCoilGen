from dataclasses import dataclass
import numpy as np


def calculate_force_and_torque_matrix(coil_parts, gauss_order, conductor_thickness, specific_conductivity_copper):
    """
    Calculate the force and torque matrix for each coil part.

    Args:
        coil_parts (list): List of CoilPart objects representing different parts of the coil.
        gauss_order (int): Gauss order for numerical integration.
        conductor_thickness (float): Thickness of the conductor.
        specific_conductivity_copper (float): Specific conductivity of copper.

    Returns:
        list: Updated list of CoilPart objects with force and torque matrices calculated.
    """
    material_factor = specific_conductivity_copper / conductor_thickness

    for part in coil_parts:
        num_nodes = len(part.basis_elements)

        # Calculate the adjacency matrix to mark mesh node neighbors
        node_adjacency_mat = np.zeros((num_nodes, num_nodes), dtype=bool)
        for tri_ind in range(part.coil_mesh.faces.shape[1]):
            node_adjacency_mat[part.coil_mesh.faces[0, tri_ind],
                               part.coil_mesh.faces[1, tri_ind]] = True
            node_adjacency_mat[part.coil_mesh.faces[1, tri_ind],
                               part.coil_mesh.faces[2, tri_ind]] = True
            node_adjacency_mat[part.coil_mesh.faces[2, tri_ind],
                               part.coil_mesh.faces[0, tri_ind]] = True

        vert1, vert2 = np.where(node_adjacency_mat)
        mesh_edges = np.column_stack((vert1, vert2))
        mesh_edges_non_unique = np.vstack((np.arange(num_nodes), mesh_edges[:, 0]),
                                          np.vstack((np.arange(num_nodes), mesh_edges[:, 1]))).T

        node_adjacency_mat = np.logical_or(
            node_adjacency_mat, node_adjacency_mat.T)
        part.node_adjacency_mat = node_adjacency_mat

        # Calculate the matrix of spatial distances for neighboring vertices
        nodal_neighbor_distances = np.linalg.norm(np.repeat(part.coil_mesh.vertices[:, :, np.newaxis],
                                                            part.coil_mesh.vertices.shape[2], axis=2)
                                                  - np.flip(np.repeat(part.coil_mesh.vertices[:, :, np.newaxis],
                                                                      part.coil_mesh.vertices.shape[2], axis=2), axis=1),
                                                  ord=2, axis=1)
        part.nodal_neighbor_distances = np.squeeze(
            nodal_neighbor_distances) * node_adjacency_mat

        resistance_matrix = np.zeros((num_nodes, num_nodes))
        for edge_ind in range(mesh_edges_non_unique.shape[0]):
            node_ind1 = mesh_edges_non_unique[edge_ind, 0]
            node_ind2 = mesh_edges_non_unique[edge_ind, 1]
            overlapping_triangles = np.intersect1d(part.basis_elements[node_ind1].triangles,
                                                   part.basis_elements[node_ind2].triangles)
            resistance_sum = 0
            if overlapping_triangles.size > 0:
                for overlapp_tri_ind in overlapping_triangles:
                    first_node_triangle_position = part.basis_elements[
                        node_ind1].triangles == overlapp_tri_ind
                    second_node_triangle_position = part.basis_elements[
                        node_ind2].triangles == overlapp_tri_ind
                    triangle_area = part.basis_elements[node_ind1].area[first_node_triangle_position]
                    primary_current = part.basis_elements[node_ind1].current[first_node_triangle_position, :]
                    secondary_current = part.basis_elements[node_ind2].current[second_node_triangle_position, :]
                    resistance_sum += np.dot(primary_current,
                                             secondary_current) * (triangle_area ** 2)
                resistance_matrix[node_ind1, node_ind2] = resistance_sum

        resistance_matrix += resistance_matrix.T
        resistance_matrix *= material_factor
        part.resistance_matrix = resistance_matrix

    return coil_parts
