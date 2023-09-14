# System imports
from typing import List
import numpy as np

# Logging
import logging

# Local imports
from .data_structures import CoilPart

log = logging.getLogger(__name__)


def calculate_one_ring_by_mesh(coil_parts: List[CoilPart]):
    """
    Calculate the one-ring neighborhood of vertices in the coil mesh.

    Initialises the following properties of a CoilPart:
        - one_ring_list
        - node_triangles
        - node_triangle_mat

    Depends on the following input_args:
        - None

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): List of coil parts.

    Returns:
        List[CoilPart]: Updated list of coil parts with calculated one ring information.
    """
    for part_ind in range(len(coil_parts)):
        # Extract the intermediate variables
        coil_part = coil_parts[part_ind]
        part_mesh = coil_part.coil_mesh  # Get the Mesh instance
        part_vertices = part_mesh.get_vertices()  # Get the vertices for the coil part.
        part_faces = part_mesh.get_faces()
        num_vertices = part_vertices.shape[0]
        # NOTE: Not calculated the same as MATLAB
        node_triangles = part_mesh.vertex_faces()

        # Extract the indices of the corners of the connected triangles
        node_triangles_corners = [
            part_faces[x, :] for x in node_triangles
        ]  # Get triangle corners for each node

        # Create array with the vertex indices of neighboring vertices for each vertex in the mesh.
        one_ring_list = np.empty(num_vertices, dtype=object)  # []
        for node_ind in range(num_vertices):
            single_cell = node_triangles_corners[node_ind]  # Arrays of corners, including current index
            neighbor_faces = np.asarray([x[x != node_ind] for x in single_cell])
            one_ring_list[node_ind] = neighbor_faces  # m (vertices) x n_m (neighbours) array of neighbouring vertices

        # Make sure that the current orientation is uniform for all elements
        for node_ind in range(num_vertices):
            for face_ind in range(len(one_ring_list[node_ind])):
                point_aa = part_vertices[one_ring_list[node_ind][face_ind][0]]
                point_bb = part_vertices[one_ring_list[node_ind][face_ind][1]]
                point_cc = part_vertices[node_ind]
                cross_vec = np.cross(point_bb - point_aa, point_aa - point_cc)

                if np.sign(np.dot(part_mesh.n[node_ind], cross_vec)) > 0:
                    one_ring_list[node_ind][face_ind] = np.flipud(
                        one_ring_list[node_ind][face_ind]
                    )

        # Update the coil_part with one-ring neighborhood information
        node_triangle_mat = np.zeros((num_vertices, part_faces.shape[0]), dtype=int)

        coil_part.one_ring_list = one_ring_list
        coil_part.node_triangles = node_triangles
        coil_part.node_triangle_mat = node_triangle_mat

    return coil_parts
