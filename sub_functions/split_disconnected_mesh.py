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

    # Initialise variables
    face_set_vertices = np.full((num_vertices), -1, dtype=int)  # Unused
    face_sets = np.ndarray((num_vertices), dtype=object)
    face_sets[0] = faces[0]  # Add the indices of the first face
    face_set_index = 0
    num_face_sets = 1

    # Check intersection of faces with face_sets
    connected_faces = []
    lists_of_connected_faces = [connected_faces]
    for i in range(0, num_faces):
        merge = False
        for found_face_set in range(num_face_sets):
            intersection = np.intersect1d(face_sets[found_face_set], faces[i])
            if intersection.size > 0:
                merge = True
                break

        if merge:
            # Merge the sets if faces have a common vertex
            face_sets[found_face_set] = np.union1d(face_sets[found_face_set], faces[i])
            face_set_vertices[faces[i]] = found_face_set
            connected_faces.append(faces[i])
        else:
            # log.debug(" %s not in %s", faces[i], face_sets)
            # Start new connected list
            num_face_sets += 1
            face_set_index += 1
            face_sets[face_set_index] = faces[i]
            face_set_vertices[faces[i]] = face_set_index
            # Create and add current connected_faces to running list
            connected_faces = [faces[i]]
            lists_of_connected_faces.append(connected_faces)

    return lists_of_connected_faces, face_set_vertices


def split_disconnected_mesh_old(coil_mesh_in: Mesh) -> List[CoilPart]:
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
        face_connections_x = face_connections - np.min(face_connections)
        mesh_part = Mesh(vertices=connected_vertices, faces=face_connections_x)
        mesh_part.unique_vert_inds = face_connections
        mesh_part.normal_rep = coil_mesh_in.normal_rep
        coil_parts.append(CoilPart(coil_mesh=mesh_part))
        index += 1

    return coil_parts

def split_disconnected_mesh(coil_mesh_in: Mesh) -> List[CoilPart]:
    """
    Split the mesh and the stream function if there are disconnected pieces such as shielded gradients.

    Args:
        coil_mesh_in (Mesh): Input mesh object.

    Returns:
        List[Mesh]: List of coil parts with split mesh.

    """
    tri_mesh = coil_mesh_in.trimesh_obj

    if tri_mesh.body_count > 1:
        coil_parts = [None] * tri_mesh.body_count
        sub_meshes = tri_mesh.split(only_watertight=False)
        index = 0
        for sub_mesh in sub_meshes:
            mesh_part = Mesh(trimesh_obj=sub_mesh)
            mesh_part.normal_rep = coil_mesh_in.normal_rep
            coil_parts[index] = CoilPart(coil_mesh=mesh_part)
            index += 1
        return coil_parts
    else:
        return [CoilPart(coil_mesh=coil_mesh_in)]
