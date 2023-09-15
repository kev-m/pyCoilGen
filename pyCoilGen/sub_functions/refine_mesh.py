# System imports
import numpy as np

# Logging
import logging

log = logging.getLogger(__name__)


def refine_mesh(coil_parts, input_args):
    """
    Increase the resolution of the mesh by splitting every existing face into four.

    Args:
        coil_parts (list): List of coil parts.
        input (dict): Input parameters.

    Returns:
        coil_parts (list): List of coil parts with refined mesh.
    """

    iteration_num_mesh_refinement = input_args.iteration_num_mesh_refinement
    sf_source_file = input_args.sf_source_file

    if sf_source_file == 'none':
        for part_ind in range(len(coil_parts)):
            coil_part = coil_parts[part_ind]
            part_mesh = coil_part.coil_mesh
            part_vertices = part_mesh.get_vertices()  # Get the vertices for the coil part.
            part_faces = part_mesh.get_faces()  # Get the faces for the coil part.
            log.debug(" %d --- part_faces.shape: %s", part_ind, part_faces.shape)

            for num_subdivision_sf in range(iteration_num_mesh_refinement-1):
                # Recreate the coil_part and let Trimesh sort out vertex and face cleanup
                new_vertices, new_faces = refine_mesh_elements1(part_mesh.get_vertices(), part_mesh.get_faces())
                part_mesh.recreate(vertices=new_vertices, faces=new_faces)

            coil_parts[part_ind] = coil_part

    return coil_parts


def refine_mesh_elements1(vertices, faces):
    # Upsample the stream function
    # Calculate edge centers
    coord_1_3 = np.mean(vertices[faces[:, [0, 1]], :], axis=1)
    coord_3_2 = np.mean(vertices[faces[:, [1, 2]], :], axis=1)
    coord_2_1 = np.mean(vertices[faces[:, [2, 0]], :], axis=1)

    # Upsample the mesh
    new_vertices = np.concatenate((vertices, coord_1_3, coord_3_2, coord_2_1))
    coord_ind_1 = faces[:, 0]
    coord_ind_2 = faces[:, 1]
    coord_ind_3 = faces[:, 2]
    new_coord_inds_1_3 = np.arange(0, coord_1_3.shape[0]) + vertices.shape[0]
    new_coord_inds_3_2 = np.arange(0, coord_3_2.shape[0]) + (vertices.shape[0] + coord_1_3.shape[0])
    new_coord_inds_2_1 = np.arange(0, coord_2_1.shape[0]) + \
        (vertices.shape[0] + coord_1_3.shape[0] + coord_3_2.shape[0])

    # Build the new triangles
    new_face_1 = np.column_stack((coord_ind_1, new_coord_inds_1_3, new_coord_inds_2_1))
    new_face_2 = np.column_stack((new_coord_inds_1_3, coord_ind_3, new_coord_inds_3_2))
    new_face_3 = np.column_stack((new_coord_inds_3_2, coord_ind_2, new_coord_inds_2_1))
    new_face_4 = np.column_stack((new_coord_inds_1_3, new_coord_inds_3_2, new_coord_inds_2_1))
    new_faces = np.concatenate((new_face_1, new_face_2, new_face_3, new_face_4))

    # Delete double counted nodes
    # all_coords_unique: The sorted unique values
    # unique_inverse: The indices to reconstruct the original array from the unique array.
    all_coords_unique, unique_inverse = np.unique(new_vertices, axis=0, return_inverse=True)
    log.debug(" --- new_faces.shape: %s", new_faces.shape)
    log.debug(" --- unique_inverse.shape: %s", unique_inverse.shape)
    new_faces = np.reshape(unique_inverse[new_faces.flatten()], new_faces.shape)

    return new_vertices, new_faces


def refine_mesh_elements2(vertices, faces):
    # Upsample the stream function
    # Calculate edge centers
    coord_1_2 = np.mean(vertices[faces[:, [0, 1]], :], axis=1)
    coord_2_3 = np.mean(vertices[faces[:, [1, 2]], :], axis=1)
    coord_3_1 = np.mean(vertices[faces[:, [2, 0]], :], axis=1)

    # Upsample the mesh
    new_vertices = np.concatenate((vertices, coord_1_2, coord_2_3, coord_3_1))
    coord_ind_1 = faces[:, 0]
    coord_ind_2 = faces[:, 1]
    coord_ind_3 = faces[:, 2]
    new_coord_inds_1_2 = np.arange(0, coord_1_2.shape[0]) + vertices.shape[0]
    new_coord_inds_2_3 = np.arange(0, coord_2_3.shape[0]) + (vertices.shape[0] + coord_1_2.shape[0])
    new_coord_inds_3_1 = np.arange(0, coord_3_1.shape[0]) + \
        (vertices.shape[0] + coord_1_2.shape[0] + coord_2_3.shape[0])

    # Build the new triangles
    new_face_1 = np.column_stack((coord_ind_1, new_coord_inds_3_1, new_coord_inds_1_2))
    new_face_2 = np.column_stack((new_coord_inds_1_2, new_coord_inds_2_3, coord_ind_3))
    new_face_3 = np.column_stack((new_coord_inds_3_1, coord_ind_2, new_coord_inds_2_3))
    new_face_4 = np.column_stack((new_coord_inds_3_1, new_coord_inds_2_3, new_coord_inds_1_2))
    new_faces = np.concatenate((new_face_1, new_face_2, new_face_3, new_face_4))

    # Delete double counted nodes
    # all_coords_unique: The sorted unique values
    # unique_inverse: The indices to reconstruct the original array from the unique array.
    all_coords_unique, unique_inverse = np.unique(new_vertices, axis=0, return_inverse=True)
    log.debug(" ---2 new_faces.shape: %s", new_faces.shape)
    log.debug(" ---2 unique_inverse.shape: %s", unique_inverse.shape)
    # new_faces = np.reshape(unique_inverse[new_faces.flatten()], new_faces.shape)
    # new_faces = np.reshape(unique_inverse[new_faces], new_faces.shape)

    return new_vertices, new_faces


def refine_mesh_delegated(coil_parts, input_args):
    """
    Increase the resolution of the mesh.

    Initialises the following properties of the CoilParts:
        - None

    Depends on the following properties of the CoilParts:
        - coil_mesh

    Depends on the following input_args:
        - sf_source_file

    Updates the following properties of a CoilPart:
        - coil_mesh

    Args:
        coil_parts (List[Mesh]): A list of Mesh objects.
        input (object): Input object with attributes 'iteration_num_mesh_refinement' and 'sf_source_file'.

    Returns:
        (List[Mesh]): Updated coil parts object with refined mesh.

    """
    iteration_num_mesh_refinement = input_args.iteration_num_mesh_refinement
    sf_source_file = input_args.sf_source_file

    if sf_source_file == 'none':
        log.debug(" - iteration_num_mesh_refinement: %d", iteration_num_mesh_refinement)
        for part_ind in range(len(coil_parts)):
            for i in range(iteration_num_mesh_refinement):
                log.debug(" - Refining part %d", part_ind)
                coil_parts[part_ind].coil_mesh.refine(inplace=True)

    return coil_parts

