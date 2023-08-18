# System imports
from typing import List
import numpy as np
# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart, Mesh

log = logging.getLogger(__name__)


def split_disconnected_mesh_trimesh(coil_mesh_in: Mesh) -> List[CoilPart]:
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
        index = len(sub_meshes)-1
        for sub_mesh in sub_meshes:
            mesh_part = Mesh(trimesh_obj=sub_mesh)
            mesh_part.normal_rep = coil_mesh_in.normal_rep
            coil_parts[index] = CoilPart(coil_mesh=mesh_part)
            index -= 1
        return coil_parts
    else:
        return [CoilPart(coil_mesh=coil_mesh_in)]


def split_disconnected_mesh(coil_mesh_in: Mesh) -> List[CoilPart]:
    """
    Split the mesh and the stream function if there are disconnected pieces
    such as shielded gradients.

    Args:
        coil_mesh_in (Mesh): Input mesh to be split.

    Returns:
        List[CoilPart]: A list of CoilPart structures representing the split mesh parts.
    """
    coil_parts = []  # Initialize a list to store the split mesh parts

    faces_in = coil_mesh_in.get_faces()

    vert_group = np.zeros((faces_in.shape[0], 1), dtype=np.uint32)
    current_group = 0

    # Group the faces according to their connectivity
    while np.any(vert_group == 0):
        current_group += 1
        next_face = np.where(vert_group == 0)[0][0]
        verts_to_sort = faces_in[next_face, :]

        while verts_to_sort.size > 0:
            availFaceInds = np.where(vert_group == 0)[0]
            availFaceSub, _ = np.where(np.isin(faces_in[availFaceInds, :], verts_to_sort))
            vert_group[availFaceInds[availFaceSub]] = current_group
            verts_to_sort = faces_in[availFaceInds[availFaceSub], :]

    # Create parts based on the discovered groups
    num_vert_groups = current_group
    for current_group in range(1, num_vert_groups + 1):
        # Extract the faces of "this" group.
        group_vert_faces = vert_group.flatten() == current_group
        faces_of_group = faces_in[group_vert_faces, :]

        # Extract the unique faces
        sortedUniqueFaces, unique_indices = np.unique(faces_of_group, return_inverse=True, axis=0)
        group_indices = np.sort(unique_indices) # Don't change the original order
        face_min = np.min(sortedUniqueFaces)
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

        part = CoilPart(coil_mesh=coil_mesh)
        coil_parts.append(part)

    return coil_parts
