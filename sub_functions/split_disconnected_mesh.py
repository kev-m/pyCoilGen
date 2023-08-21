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

    changed = True
    while changed:
        for face_index, face in enumerate(faces):
            changed = False
            # Debug
            ## if face_index == 39:
            ##    log.debug("Start now!")
            # If any vertex of face is in the current vertex group, add this face to the current face_group, etc.
            if face_group[face_index] == 0:
                for vertex in face:
                    if len(vert_group[mesh_id-1]) == 0 or vertex in vert_group[mesh_id-1]:
                        log.debug("Group: %d => Adding Face %d,  %s", mesh_id, face_index, face)
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

    # Having found all the vert_groups, now check for vert_group intersections and merge such groups
    changed = True
    while changed:
        changed = False
        for index1, group1 in enumerate(vert_group):
            for index2 in range(index1+1, len(vert_group)):
                group2 = vert_group[index2]
                if len(group1.intersection(group2)) > 0:
                    log.debug("Found intersection between groups %d and %d, merging", index1, index2)
                    group1 = group1.union(group2)
                    changed = True
                    # Assign all faces to index1
                    which = next(iter(group1))
                    group1_id = face_group[which] # ??
                    face_group[list(group2)] = group1_id
                    # Remove group2 from vert_group
                    vert_group.pop(index2)
                    break
            if changed:
                break



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
