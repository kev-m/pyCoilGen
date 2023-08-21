# System imports
from typing import List
import numpy as np
# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart, Mesh

log = logging.getLogger(__name__)


def split_disconnected_mesh(coil_mesh_in: Mesh) -> List[CoilPart]:
    """
    Split the mesh and the stream function if there are disconnected pieces
    such as shielded gradients.

    Args:
        coil_mesh_in (Mesh): Input mesh to be split.

    Returns:
        List[CoilPart]: A list of CoilPart structures representing the split mesh parts.
    """

    vertices = coil_mesh_in.get_vertices()
    faces = coil_mesh_in.get_faces()

    face_group = np.zeros((faces.shape[0]), dtype=np.int32)
    mesh_id = 1  # Initialize the mesh ID
    vert_group =[set()]

    faces_temp = faces.tolist()
    changed = True
    while changed:
        for face_index, face in enumerate(faces_temp):
            changed = False
            # Debug
            if face_index == 39:
                log.debug("Start now!")
            # If any vertex of face is in the current vertex group, add this face to the current face_group, etc.
            if face_group[face_index] == 0:
                for vertex in face:
                    if len(vert_group[mesh_id-1]) == 0 or vertex in vert_group[mesh_id-1]:
                        log.debug("Group: %d => Face %d, adding vertex %d", mesh_id, face_index, vertex)
                        changed = True
                        break

            if changed:
                # Add vertices to list
                for vertex in face:
                    vert_group[mesh_id-1].add(vertex)
                # Mark this face as processed
                face_group[face_index] = mesh_id


        if changed == False and np.any(face_group == 0):
            log.debug("Starting new group: %d", mesh_id+1)
            mesh_id += 1
            vert_group.append(set())
            changed = True



    # Initialize a list to store the split mesh parts
    coil_parts = []
    # Create parts based on the discovered groups
    num_vert_groups = mesh_id
    for current_group in range(1, num_vert_groups + 1):
        # Extract the faces of "this" group.
        group_vert_faces = face_group.flatten() == current_group
        faces_of_group = faces[group_vert_faces, :]

        # Extract the unique faces
        sortedUniqueFaces, unique_indices = np.unique(faces_of_group, return_inverse=True, axis=0)
        face_min = np.min(sortedUniqueFaces[0]) # Lowest vertex ID
        group_indices = np.sort(unique_indices)  # Don't change the original order
        uniqueFaces = faces_of_group[group_indices]

        # Create a new CoilPart structure for each split mesh part
        vertices_in = coil_mesh_in.get_vertices()
        # Now extract the used vertex indices from the unique faces.
        uniqueVertIndices = np.unique(sortedUniqueFaces.flatten(), axis=0)
        uniqueVerts = vertices_in[uniqueVertIndices, :]

        # Create the new Mesh and add it to the coil_parts list.
        coil_mesh = Mesh(faces=uniqueFaces-face_min, vertices=uniqueVerts)
        coil_mesh.unique_vert_inds = uniqueVertIndices
        coil_mesh.normal_rep = coil_mesh_in.normal_rep

        coil_parts.append(CoilPart(coil_mesh))

    return coil_parts
