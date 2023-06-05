# System imports
import numpy as np
# Logging
import logging

# Local imports
from data_structures import DataStructure

log = logging.getLogger(__name__)


def split_disconnected_mesh(coil_mesh_in):
    """
    Split the mesh and the stream function if there are disconnected pieces, such as shielded gradients.

    Args:
        coil_mesh_in (object): Input coil mesh object with attributes 'faces' and 'vertices'.

    Returns:
        coil_parts (list): List of coil parts, each containing a separate mesh.

    """

    coil_mesh_in.faces = coil_mesh_in.faces#.T  # Transpose the faces array

    # Initialize vertex group array
    vert_group = np.zeros(coil_mesh_in.faces.shape[0], dtype=np.uint32)
    current_group = 0  # Initialize current group counter

    while np.any(vert_group == 0):
        current_group += 1
        # Find the next ungrouped face
        next_face = np.where(vert_group == 0)[0][0]
        verts_to_sort = coil_mesh_in.faces[next_face, :]

        while verts_to_sort.size > 0:
            availFaceInds = np.where(vert_group == 0)[0]
            availFaceSub, _ = np.where(
                np.isin(coil_mesh_in.faces[availFaceInds, :], verts_to_sort))
            vert_group[availFaceInds[availFaceSub]] = current_group
            verts_to_sort = coil_mesh_in.faces[availFaceInds[availFaceSub], :]

    num_vert_groups = current_group
    
    coil_parts = [None] * num_vert_groups  # Create a list to store coil parts

    for current_group in range(1, num_vert_groups + 1):
        faces_of_group = coil_mesh_in.faces[vert_group == current_group, :]

        log.debug("Shape: %s, %s", faces_of_group.shape,
                  coil_mesh_in.vertices.shape)

        unqVertIds, _, newVertIndices = np.unique(
            faces_of_group, return_index=True, return_inverse=True)
        unqVertIds -= 1 # Matlab uses 1 index, so must subtract 1 from face indices
        faces = np.reshape(newVertIndices, faces_of_group.shape)#.T
        vertices = coil_mesh_in.vertices[unqVertIds, :]
        coil_part = DataStructure(
            faces=faces, vertices=vertices, unique_vert_inds=unqVertIds)
        coil_parts[current_group - 1] = DataStructure(coil_mesh=coil_part)

    return coil_parts
