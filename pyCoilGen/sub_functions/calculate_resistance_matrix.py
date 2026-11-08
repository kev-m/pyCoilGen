# System imports
from typing import List
import numpy as np

# Logging
import logging


# Local imports
from .constants import get_level, DEBUG_VERBOSE
from .data_structures import CoilPart

log = logging.getLogger(__name__)


def _node_pair_resistance(basis_elements, num_faces: int, num_nodes: int) -> np.ndarray:
    """
    For every pair of nodes (n1, n2), the sum over the triangles that contain both nodes of the dot product of their
    current densities times the squared area of the triangle, as seen from n1. This includes the pairs (n, n).

    Every triangle has three corner nodes, so it contributes to nine node pairs. The contributions are added in the order
    of the triangle index, as the original loop over the shared triangles of each pair did.

    Args:
        basis_elements (list): The basis element of every node.
        num_faces (int): The number of faces of the mesh.
        num_nodes (int): The number of nodes of the mesh.

    Returns:
        ndarray: The sums, (num nodes, num nodes).
    """
    counts = np.array([len(element.area) for element in basis_elements], dtype=int)
    node_of_entry = np.repeat(np.arange(num_nodes), counts)
    non_empty = [element for element in basis_elements if len(element.area)]
    face_of_entry = np.concatenate([np.asarray(element.triangles)[:len(element.area)] for element in non_empty]).astype(int)
    area = np.concatenate([np.reshape(element.area, (-1,)) for element in non_empty])
    current = np.concatenate([np.reshape(element.current, (-1, 3)) for element in non_empty])

    order = np.argsort(face_of_entry, kind='stable')  # by face; the order of the entries of a face is not important
    node_of_entry, face_of_entry, area, current = node_of_entry[order], face_of_entry[order], area[order], current[order]
    entries_per_face = np.bincount(face_of_entry, minlength=num_faces)
    if len(face_of_entry) != 3 * num_faces or np.any(entries_per_face != 3):
        raise ValueError("Every face must be a triangle with three nodes, each with a basis element that contains it.")

    # The three entries of every face: (faces, 3 corners, ...)
    nodes = node_of_entry.reshape(num_faces, 3)
    areas = area.reshape(num_faces, 3)
    currents = current.reshape(num_faces, 3, 3)
    # contribution[face, p, q] = (current_p . current_q) * area_p^2, for the nodes p and q of the face
    contribution = np.einsum('fpx,fqx->fpq', currents, currents) * (areas ** 2)[:, :, np.newaxis]
    rows = np.repeat(nodes[:, :, np.newaxis], 3, axis=2)
    cols = np.repeat(nodes[:, np.newaxis, :], 3, axis=1)

    resistance = np.zeros((num_nodes, num_nodes))
    np.add.at(resistance, (rows.ravel(), cols.ravel()), contribution.ravel())  # unbuffered, in the order of the faces
    return resistance


def calculate_resistance_matrix(coil_parts: List[CoilPart], input_args) -> List[CoilPart]:
    """
    Calculate the resistance matrix for coil parts.

    Initialises the following properties of a CoilPart:
        - resistance_matrix
        - node_adjacency_mat

    Depends on the following properties of the CoilParts:
        - basis_elements

    Depends on the following input_args:
        - conductor_thickness
        - specific_resistivity_conductor

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (list): List of coil parts.
        input: The input parameters.

    Returns:
        coil_parts (List[CoilPart]): Updated list of coil parts with resistance matrix.

    """

    conductor_thickness = input_args.conductor_thickness
    conductor_resistivity = input_args.specific_resistivity_conductor
    material_factor = conductor_resistivity / conductor_thickness

    for part_ind in range(len(coil_parts)):
        # Setup variables
        coil_part = coil_parts[part_ind]
        part_mesh = coil_part.coil_mesh
        part_faces = part_mesh.get_faces()  # Get the faces for this mesh
        num_faces = part_faces.shape[0]

        num_nodes = len(coil_part.basis_elements)  # Same as number of vertices

        # Calculate node adjacency matrix: a directed edge for each side of each face (row: second node, column: first node)
        node_adjacency_mat = np.zeros((num_nodes, num_nodes), dtype=bool)
        node_adjacency_mat[part_faces[:, 1], part_faces[:, 0]] = True
        node_adjacency_mat[part_faces[:, 2], part_faces[:, 1]] = True
        node_adjacency_mat[part_faces[:, 0], part_faces[:, 2]] = True

        # The resistance matrix is evaluated for the nodes and the directed edges (first node, second node)
        edge_mask = node_adjacency_mat.T.copy()
        edge_mask[np.arange(num_nodes), np.arange(num_nodes)] = True

        node_adjacency_mat = np.logical_or(node_adjacency_mat, node_adjacency_mat.T)
        coil_part.node_adjacency_mat = node_adjacency_mat

        # Calculate resistance matrix
        resistance_matrix = _node_pair_resistance(coil_part.basis_elements, num_faces, num_nodes)
        resistance_matrix[~edge_mask] = 0
        resistance_matrix += resistance_matrix.T
        resistance_matrix *= material_factor

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" resistance_matrix shape: %s", resistance_matrix.shape)

        coil_part.resistance_matrix = resistance_matrix

    return coil_parts
