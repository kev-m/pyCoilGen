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

        # The triangles of all nodes, flattened: triangle k belongs to node `node_of_triangle[k]`
        triangles_per_node = np.array([len(x) for x in node_triangles], dtype=int)
        node_of_triangle = np.repeat(np.arange(num_vertices), triangles_per_node)
        triangle_corners = part_faces[np.concatenate([np.asarray(x, dtype=int) for x in node_triangles])]             if triangles_per_node.sum() else np.zeros((0, 3), dtype=int)  # The corners, including the node itself

        # The two other corners of each triangle are the neighbouring vertices in the one ring of the node
        other_corners = triangle_corners != node_of_triangle[:, np.newaxis]
        if not np.all(other_corners.sum(axis=1) == 2):
            raise ValueError("Every triangle of a node must have exactly two corners other than the node.")
        neighbours = triangle_corners[other_corners].reshape(-1, 2)

        # Make sure that the current orientation is uniform for all elements
        point_aa = part_vertices[neighbours[:, 0]]
        point_bb = part_vertices[neighbours[:, 1]]
        point_cc = part_vertices[node_of_triangle]
        cross_vec = np.cross(point_bb - point_aa, point_aa - point_cc)
        flip = np.sign(np.sum(part_mesh.n[node_of_triangle] * cross_vec, axis=1)) > 0
        neighbours[flip] = neighbours[flip, ::-1]

        # Create array with the vertex indices of neighboring vertices for each vertex in the mesh.
        one_ring_list = np.empty(num_vertices, dtype=object)  # []
        first_triangle = np.cumsum(triangles_per_node) - triangles_per_node
        for node_ind in range(num_vertices):
            if triangles_per_node[node_ind]:
                first = first_triangle[node_ind]
                # m (vertices) x n_m (neighbours) array of neighbouring vertices
                one_ring_list[node_ind] = neighbours[first:first + triangles_per_node[node_ind]].copy()
            else:
                one_ring_list[node_ind] = np.asarray([])

        # Update the coil_part with one-ring neighborhood information
        node_triangle_mat = np.zeros((num_vertices, part_faces.shape[0]), dtype=int)

        coil_part.one_ring_list = one_ring_list
        coil_part.node_triangles = node_triangles
        coil_part.node_triangle_mat = node_triangle_mat

    return coil_parts
