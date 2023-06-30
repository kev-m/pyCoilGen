# System imports
from typing import List
import numpy as np
# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart, Mesh

log = logging.getLogger(__name__)


def compute_connected_face_indices(vertices, faces):
    num_vertices = len(vertices)
    num_faces = len(faces)
    connected_indices = []

    # Initialise variables
    face_set_vertices = np.full((num_vertices), -1, dtype=int)  # Unused
    face_sets = np.ndarray((num_vertices), dtype=object)
    face_sets[0] = faces[0]  # Add the indices of the first face
    face_set_index = 0

    # Check intersection of faces with face_sets
    connected_faces = []
    lists_of_connected_faces = [connected_faces]
    for i in range(0, num_faces-1):
        intersection = np.intersect1d(face_sets[face_set_index], faces[i])
        if intersection.size > 0:
            # Merge the sets if faces have a common vertex
            face_sets[face_set_index] = np.union1d(face_sets[face_set_index], faces[i])
            face_set_vertices[faces[i]] = face_set_index
            connected_faces.append(faces[i])
        else:
            print(faces[i], "not in", face_sets)
            face_set_index += 1
            # face_sets[face_set_index] = faces[i]
            face_sets = np.append(face_sets, faces[i])
            face_set_vertices[faces[i]] = face_set_index
            lists_of_connected_faces.append(connected_faces)  # Add old list
            connected_faces = [faces[i]]
            lists_of_connected_faces.append(connected_faces)  # Add new list

    return lists_of_connected_faces, face_set_vertices


def split_disconnected_mesh(coil_mesh_in: Mesh) -> List[CoilPart]:
    """
    Split the mesh and the stream function if there are disconnected pieces such as shielded gradients.

    Args:
        coil_mesh_in (Mesh): Input mesh object.

    Returns:
        List[Mesh]: List of coil parts with split mesh.

    """
    mesh_faces = coil_mesh_in.get_faces()  # Each row is a face, the column is the vertex index
    mesh_vertices = coil_mesh_in.get_vertices()

    connected_faces, face_set_vertices = compute_connected_face_indices(vertices=mesh_vertices, faces=mesh_faces)
    coil_parts = []
    index = 0
    for face_connections in connected_faces:
        vertex_indices = np.where(face_set_vertices == index)[0]
        connected_vertices = mesh_vertices[vertex_indices, :]
        mesh_part = Mesh(vertices=connected_vertices, faces=face_connections)
        mesh_part.unique_vert_inds = face_connections
        mesh_part.normal_rep = coil_mesh_in.normal_rep
        coil_parts.append(CoilPart(coil_mesh=mesh_part))
        index += 1

    return coil_parts
