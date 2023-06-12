import numpy as np
from scipy.spatial import Delaunay

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilSolution

log = logging.getLogger(__name__)

def calculate_one_ring_by_mesh(coil_solution : CoilSolution, coil_parts, input):
    """
    Calculate the one-ring neighborhood of vertices in the coil mesh.

    Args:
        coil_parts (list): List of CoilPart objects.
        input (object): Input object containing evaluation parameters.

    Returns:
        list: Updated coil_parts with one-ring neighborhood information.
    """
    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        part_mesh = coil_part.coil_mesh
        part_vertices = part_mesh.get_vertices()
        part_faces = part_mesh.get_faces()
        trimesh_obj = part_mesh.trimesh_obj
        optimisation = coil_solution.optimisation

        if not optimisation.use_preoptimization_temp:
            num_nodes = part_vertices.shape[0] # The number of vertices
            # Extract the corner vertices of triangles for each node
            node_triangles = trimesh_obj.vertex_adjacency_graph
            node_triangles_corners = [part_faces[node_triangles[node_ind]] for node_ind in range(num_nodes)]
            one_ring_list = [None] * num_nodes

            # Compute the one-ring neighborhood for each vertex
            for node_ind in range(num_nodes):
                single_cell = [node_triangles_corners[node_ind][i].tolist() for i in range(len(node_triangles_corners[node_ind]))]
                one_ring_list[node_ind] = [cell for cell in single_cell if node_ind not in cell]

            # Make sure that the current orientation is uniform for all elements
            for node_ind in range(num_nodes):
                for face_ind in range(len(one_ring_list[node_ind])):
                    point_aa = part_vertices[one_ring_list[node_ind][face_ind][0], :]
                    point_bb = part_vertices[one_ring_list[node_ind][face_ind][1], :]
                    point_cc = part_vertices[node_ind, :]
                    cross_vec = np.cross(point_bb - point_aa, point_aa - point_cc)

                    if np.dot(coil_part.coil_mesh.n[node_ind, :], cross_vec) > 0:
                        one_ring_list[node_ind][face_ind] = np.flipud(one_ring_list[node_ind][face_ind])

            node_triangle_mat = np.zeros((num_nodes, part_faces.shape[0]), dtype=bool)

            coil_part.one_ring_list = one_ring_list
            coil_part.node_triangles = node_triangles
            coil_part.node_triangle_mat = node_triangle_mat

        else:
            raise Exception("Optimisation is not implemented!")
            # Use pre-optimized data from input object
            coil_part.one_ring_list = optimisation.one_ring_list
            coil_part.node_triangles = optimisation.node_triangles
            coil_part.node_triangle_mat = optimisation.node_triangle_mat

    return coil_parts

    """
            node_triangles_corners = [
                part_faces[node_triangles[node_ind]].tolist()
                for node_ind in range(len(node_triangles))
            ]

            one_ring_list = [[] for _ in range(num_nodes)]

            # Compute the one-ring neighborhood for each vertex
            for node_ind in range(num_nodes):
                triangles = vertex_to_triangles[node_ind]
                node_triangles_corners = part_faces[triangles]
                one_ring_list[node_ind] = node_triangles_corners[:, node_triangles_corners != node_ind].T

            # Ensure consistent orientation within the one-ring neighborhood
            for node_ind in range(num_nodes):
                for face_ind in range(one_ring_list[node_ind].shape[1]):
                    point_aa = part_vertices[:, one_ring_list[node_ind][0, face_ind]]
                    point_bb = part_vertices[:, one_ring_list[node_ind][1, face_ind]]
                    point_cc = part_vertices[:, node_ind]
                    cross_vec = np.cross(point_bb - point_aa, point_aa - point_cc)

                    if np.dot(part_mesh.n[:, node_ind], cross_vec) > 0:
                        one_ring_list[node_ind][:, face_ind] = np.flipud(one_ring_list[node_ind][:, face_ind])

            node_triangle_mat = np.zeros((num_nodes, part_faces.shape[0]), dtype=bool)

            # Update the coil_parts object
            coil_part.one_ring_list = one_ring_list
            coil_part.node_triangles = vertex_to_triangles
            coil_part.node_triangle_mat = node_triangle_mat
            part_faces = part_faces.T
    """

