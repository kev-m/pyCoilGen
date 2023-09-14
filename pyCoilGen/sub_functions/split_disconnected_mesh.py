# System imports
from typing import List
import numpy as np
# Logging
import logging

# Local imports
from .data_structures import CoilPart, Mesh

log = logging.getLogger(__name__)


def split_disconnected_mesh(coil_mesh_in: Mesh) -> List[CoilPart]:
    """
    Split the mesh and the stream function if there are disconnected pieces
    such as shielded gradients.

    Initialises the following properties of the CoilParts:
        - None

    Depends on the following properties of the CoilParts:
        - None

    Depends on the following input_args:
        - None

    Args:
        coil_mesh_in (Mesh): Input mesh to be split.

    Returns:
        List[CoilPart]: A list of CoilPart structures representing the split mesh parts.
    """

    vertices = coil_mesh_in.get_vertices()
    faces = coil_mesh_in.get_faces()

    face_group = np.zeros((faces.shape[0]), dtype=np.int32)
    mesh_id = 1  # Initialize the mesh ID
    vert_groups = {mesh_id: set()}

    changed = True
    while changed:
        for face_index, face in enumerate(faces):
            changed = False
            # Debug
            # if face_index == 39:
            # log.debug("Start now!")
            # If any vertex of face is in the current vertex group, add this face to the current face_group, etc.
            if face_group[face_index] == 0:
                for vertex in face:
                    if len(vert_groups[mesh_id]) == 0 or vertex in vert_groups[mesh_id]:
                        changed = True
                        break

            if changed:
                # Add vertices to list
                for vertex in face:
                    vert_groups[mesh_id].add(vertex)
                # Mark this face as processed
                face_group[face_index] = mesh_id

        if changed == False and np.any(face_group == 0):
            mesh_id += 1
            vert_groups[mesh_id] = set()
            changed = True

    # Having found all the vert_groups, now check for vert_group intersections and merge such groups
    changed = True
    while changed:
        changed = False
        for index1, group1 in vert_groups.items():
            # for index2 in range(len(vert_groups)-1, index1, -1):
            for index2, group2 in vert_groups.items():
                if index1 == index2:
                    continue
                group2 = vert_groups[index2]
                if len(group1.intersection(group2)) > 0:
                    group1 |= group2  # Merge in place, to also update the vert_groups entry!
                    changed = True

                    # Assign all faces of group2 (index2) to group1 (index1)
                    face_group[face_group == index2] = index1

                    # Remove group2 from vert_group
                    del vert_groups[index2]
                    break
            if changed:
                break

    # Initialize a list to store the split mesh parts
    coil_parts = []
    # Create parts based on the discovered groups
    for mesh_id, mesh_group in vert_groups.items():
        # Extract the faces of "this" group.
        group_vert_faces = face_group.flatten() == mesh_id
        faces_of_group = faces[group_vert_faces, :]

        # Extract the unique faces
        sortedUniqueFaces, unique_indices = np.unique(faces_of_group, return_index=True, axis=0)
        face_min = np.min(sortedUniqueFaces[0])  # Lowest vertex ID
        group_indices = np.sort(unique_indices)  # Don't change the original order
        uniqueFaces = faces_of_group[group_indices]

        # Create a new CoilPart structure for each split mesh part
        vertices_in = coil_mesh_in.get_vertices()
        # Now extract the used vertex indices from the unique faces.
        uniqueVertIndices = np.unique(sortedUniqueFaces.flatten(), axis=0)
        uniqueVerts = vertices_in[uniqueVertIndices, :]

        if np.max(sortedUniqueFaces)-face_min >= len(uniqueVerts):
            # The unique_indices has gaps. The 'faces' array contains IDs that are larger than the number of vertices.
            # Thus the mesh is invalid.
            log.debug("Faces need be adjusted")

            true_index = 0
            for apparent_index in uniqueVertIndices:
                if apparent_index != true_index:
                    uniqueFaces[uniqueFaces == apparent_index] = true_index

                true_index += 1

            face_min = 0

        # Create the new Mesh and add it to the coil_parts list.
        coil_mesh = Mesh(faces=uniqueFaces-face_min, vertices=uniqueVerts)
        coil_mesh.unique_vert_inds = uniqueVertIndices
        coil_mesh.normal_rep = coil_mesh_in.normal_rep

        coil_parts.append(CoilPart(coil_mesh))

    return coil_parts
