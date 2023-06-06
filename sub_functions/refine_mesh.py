import numpy as np

# Logging
import logging

log = logging.getLogger(__name__)

def refine_mesh(coil_parts, input):
    """
    Increase the resolution of the mesh and interpolate the stream function.

    Args:
        coil_parts (object): Coil parts object with attributes 'coil_mesh'.
        input (object): Input object with attributes 'iteration_num_mesh_refinement' and 'sf_source_file'.

    Returns:
        coil_parts (object): Updated coil parts object with refined mesh.

    """
    iteration_num_mesh_refinement = input.iteration_num_mesh_refinement
    sf_source_file = input.sf_source_file

    if sf_source_file == 'none':
        for part_ind in range(len(coil_parts)):
            subdivided_mesh = coil_parts[part_ind].coil_mesh
            subdivided_mesh.faces = subdivided_mesh.faces.T
            subdivided_mesh.vertices = subdivided_mesh.vertices.T

            for num_subdivision_sf in range(iteration_num_mesh_refinement):
                # Upsample the stream function
                coord_1_3 = np.vstack(
                    [
                        np.mean(subdivided_mesh.vertices[subdivided_mesh.faces[:, [0, 2]]], axis=1),
                        np.mean(subdivided_mesh.vertices[subdivided_mesh.faces[:, [0, 2]]], axis=1),
                        np.mean(subdivided_mesh.vertices[subdivided_mesh.faces[:, [0, 2]]], axis=1),
                    ]
                ).T
                coord_3_2 = np.vstack(
                    [
                        np.mean(subdivided_mesh.vertices[subdivided_mesh.faces[:, [2, 1]]], axis=1),
                        np.mean(subdivided_mesh.vertices[subdivided_mesh.faces[:, [2, 1]]], axis=1),
                        np.mean(subdivided_mesh.vertices[subdivided_mesh.faces[:, [2, 1]]], axis=1),
                    ]
                ).T
                coord_2_1 = np.vstack(
                    [
                        np.mean(subdivided_mesh.vertices[subdivided_mesh.faces[:, [1, 0]]], axis=1),
                        np.mean(subdivided_mesh.vertices[subdivided_mesh.faces[:, [1, 0]]], axis=1),
                        np.mean(subdivided_mesh.vertices[subdivided_mesh.faces[:, [1, 0]]], axis=1),
                    ]
                ).T

                all_coords = np.vstack(
                    [
                        subdivided_mesh.vertices,
                        coord_1_3,
                        coord_3_2,
                        coord_2_1,
                    ]
                )
                coord_ind_1 = subdivided_mesh.faces[:, 0]
                coord_ind_2 = subdivided_mesh.faces[:, 1]
                coord_ind_3 = subdivided_mesh.faces[:, 2]
                new_coord_inds_1_3 = np.arange(1, coord_1_3.shape[1] + 1) + subdivided_mesh.vertices.shape[1]
                new_coord_inds_3_2 = np.arange(1, coord_3_2.shape[1] + 1) + (
                    subdivided_mesh.vertices.shape[1] + coord_1_3.shape[1]
                )
                new_coord_inds_2_1 = np.arange(1, coord_2_1.shape[1] + 1) + (
                    subdivided_mesh.vertices.shape[1] + coord_1_3.shape[1] + coord_3_2.shape[1]
                )

                new_tri_1 = np.column_stack([coord_ind_1, new_coord_inds_1_3, new_coord_inds_2_1])
                new_tri_2 = np.column_stack([new_coord_inds_1_3, coord_ind_3, new_coord_inds_3_2])
                new_tri_3 = np.column_stack([new_coord_inds_3_2, coord_ind_2, new_coord_inds_2_1])
                new_tri_4 = np.column_stack([new_coord_inds_1_3, new_coord_inds_3_2, new_coord_inds_2_1])
                new_tri = np.vstack([new_tri_1, new_tri_2, new_tri_3, new_tri_4])

                all_coords_unique, ic = np.unique(all_coords, axis=0, return_inverse=True)
                ind_replace_list = np.column_stack([np.arange(1, ic.shape[0] + 1), ic])

                to_replace, by_what = np.isin(new_tri, ind_replace_list[:, 0], assume_unique=True), np.isin(
                    ind_replace_list[:, 0], new_tri, assume_unique=True
                )
                new_tri[to_replace] = ind_replace_list[by_what[to_replace], 1]

                subdivided_mesh.faces = new_tri
                subdivided_mesh.vertices = all_coords_unique

            coil_parts[part_ind].coil_mesh.vertices = subdivided_mesh.vertices.T
            coil_parts[part_ind].coil_mesh.faces = subdivided_mesh.faces.T

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
            coil_parts[part_ind].coil_mesh.refine(inplace=True)

    return coil_parts
