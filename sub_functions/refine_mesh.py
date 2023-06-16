import numpy as np

# Logging
import logging

# Local imports
from sub_functions.data_structures import Mesh

log = logging.getLogger(__name__)

import numpy as np

def refine_mesh(coil_parts, input_args):
    """
    Increase the resolution of the mesh and interpolate the stream function.

    Args:
    - coil_parts (list): List of coil parts.
    - input (dict): Input parameters.

    Returns:
    - coil_parts (list): List of coil parts with refined mesh.
    """

    iteration_num_mesh_refinement = input_args['iteration_num_mesh_refinement']
    sf_source_file = input_args['sf_source_file']

    if sf_source_file == 'none':
        for part_ind in range(len(coil_parts)):
            coil_part = coil_parts[part_ind]
            part_mesh = coil_part.coil_mesh
            part_vertices = part_mesh.get_vertices()  # Get the vertices for the coil part.
            part_faces = part_mesh.get_faces()  # Get the faces for the coil part.
            log.debug(" %d --- part_faces.shape: %s", part_ind, part_faces.shape)

            for num_subdivision_sf in range(iteration_num_mesh_refinement):
                # Upsample the stream function
                # Calculate edge centers
                coord_1_3 = np.mean(part_vertices[part_faces[:, [0, 2]], :], axis=1)
                coord_3_2 = np.mean(part_vertices[part_faces[:, [2, 1]], :], axis=1)
                coord_2_1 = np.mean(part_vertices[part_faces[:, [1, 0]], :], axis=1)

                # Upsample the mesh
                all_coords = np.concatenate((part_vertices, coord_1_3, coord_3_2, coord_2_1))
                coord_ind_1 = part_faces[:, 0]
                coord_ind_2 = part_faces[:, 1]
                coord_ind_3 = part_faces[:, 2]
                new_coord_inds_1_3 = np.arange(1, coord_1_3.shape[0] + 1) + part_vertices.shape[0]
                new_coord_inds_3_2 = np.arange(1, coord_3_2.shape[0] + 1) + (part_vertices.shape[0] + coord_1_3.shape[0])
                new_coord_inds_2_1 = np.arange(1, coord_2_1.shape[0] + 1) + (part_vertices.shape[0] + coord_1_3.shape[0] + coord_3_2.shape[0])

                # Build the new triangles
                new_tri_1 = np.column_stack((coord_ind_1, new_coord_inds_1_3, new_coord_inds_2_1))
                new_tri_2 = np.column_stack((new_coord_inds_1_3, coord_ind_3, new_coord_inds_3_2))
                new_tri_3 = np.column_stack((new_coord_inds_3_2, coord_ind_2, new_coord_inds_2_1))
                new_tri_4 = np.column_stack((new_coord_inds_1_3, new_coord_inds_3_2, new_coord_inds_2_1))
                new_faces = np.concatenate((new_tri_1, new_tri_2, new_tri_3, new_tri_4))

                # Delete double counted nodes
                """
                # all_coords_unique: The sorted unique values
                # unique_inverse: The indices to reconstruct the original array from the unique array.
                all_coords_unique, unique_inverse = np.unique(all_coords, axis=0, return_inverse=True)
                log.debug(" %d --- new_faces.shape: %s", part_ind, new_faces.shape)
                log.debug(" %d --- unique_inverse.shape: %s", part_ind, unique_inverse.shape)
                ## new_faces = np.reshape(unique_inverse[new_faces.flatten()], new_faces.shape)
                new_faces = np.reshape(unique_inverse[new_faces], new_faces.shape)

                # Update the coil part mesh
                coil_part.coil_mesh.set_vertices(all_coords_unique)
                coil_part.coil_mesh.set_faces(new_faces)
                """
                # Recreate the coil_part and let Trimesh sort out vertex and face cleanup
                #coil_part = Mesh(vertices=all_coords, faces=new_faces)
                part_mesh.recreate(vertices=all_coords, faces=new_faces)

            coil_parts[part_ind] = coil_part

    return coil_parts

def refine_mesh_gpt(coil_parts, input):
    """
    Increase the resolution of the mesh.

    Args:
        coil_parts (object): Coil parts object with attributes 'coil_mesh'.
        input (object): Input object with attributes 'iteration_num_mesh_refinement' and 'sf_source_file'.

    Returns:
        coil_parts (object): Updated coil parts object with refined mesh.

    """
    iteration_num_mesh_refinement = input.iteration_num_mesh_refinement
    sf_source_file = input.sf_source_file

    if sf_source_file == 'none':
        log.debug(" - iteration_num_mesh_refinement: %d", iteration_num_mesh_refinement)
        for part_ind in range(len(coil_parts)):
            subdivided_mesh = coil_parts[part_ind].coil_mesh
            
            vertices = subdivided_mesh.vertices
            faces = subdivided_mesh.faces

            for _ in range(iteration_num_mesh_refinement):
                new_faces = []
                new_vertices = np.copy(vertices)

                for face in faces:
                    v1, v2, v3 = face

                    # Compute midpoints of the edges
                    v12 = (vertices[v1] + vertices[v2]) / 2.0
                    v23 = (vertices[v2] + vertices[v3]) / 2.0
                    v31 = (vertices[v3] + vertices[v1]) / 2.0

                    # Add new vertices to the mesh
                    new_vertices = np.vstack([new_vertices, v12, v23, v31])
                    v12_idx = len(new_vertices) - 3
                    v23_idx = len(new_vertices) - 2
                    v31_idx = len(new_vertices) - 1

                    # Create new faces using the new vertices
                    new_faces.append([v1, v12_idx, v31_idx])
                    new_faces.append([v12_idx, v2, v23_idx])
                    new_faces.append([v31_idx, v23_idx, v3])
                    new_faces.append([v12_idx, v23_idx, v31_idx])

                # Update vertices and faces for the next iteration
                vertices = new_vertices
                faces = np.array(new_faces)


            coil_parts[part_ind].coil_mesh.vertices = vertices
            coil_parts[part_ind].coil_mesh.faces = faces

    return coil_parts

def refine_mesh_delegated(coil_parts, input):
    """
    Increase the resolution of the mesh.

    Args:
        coil_parts (List[Mesh]): A list of Mesh objects.
        input (object): Input object with attributes 'iteration_num_mesh_refinement' and 'sf_source_file'.

    Returns:
        (List[Mesh]): Updated coil parts object with refined mesh.

    """
    iteration_num_mesh_refinement = input.iteration_num_mesh_refinement
    sf_source_file = input.sf_source_file

    if sf_source_file == 'none':
        log.debug(" - iteration_num_mesh_refinement: %d", iteration_num_mesh_refinement)
        for part_ind in range(len(coil_parts)):
            for i in range(iteration_num_mesh_refinement):
                log.debug(" - Refining part %d", part_ind)
                coil_parts[part_ind].coil_mesh.refine(inplace=True)

    return coil_parts

if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    # Example vertices and faces arrays
    vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [1, 1, 1], [0, 0, 0], [2, 2, 2]])
    faces = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5]])

    log.debug(" --- vertices.shape: %s", vertices.shape)
    log.debug(" --- faces.shape: %s", faces.shape)

    # Get unique vertices and their indices
    unique_vertices, vertex_indices = np.unique(vertices, axis=0, return_inverse=True)

    log.debug(" --- unique_vertices.shape: %s", unique_vertices.shape)
    log.debug(" --- vertex_indices.shape: %s", vertex_indices.shape)


    # Build new faces array with unique vertex indices
    new_faces = vertex_indices[faces]

    print(new_faces)