import numpy as np
from typing import List

from os import path
# Local functions
from sub_functions.data_structures import Mesh, CoilPart

# Logging
import logging

log = logging.getLogger(__name__)


def create_sweep_along_surface(coil_parts: List[CoilPart], input_args) -> List[CoilPart]:
    """
    Create a volumetric coil body by surface sweep.

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures containing coil_mesh.
        input_args: Input arguments structure.

    Returns:
        List[CoilPart]: List of CoilPart structures with modified properties.
    """
    convolutional_vector_length = 1  # For smoothing the curvature along the track

    if not input_args.skip_sweep:

        for part_ind in range(len(coil_parts)):

            # Define the cross section of the conductor
            if all(input_args.cross_sectional_points == 0):
                circular_resolution = 10
                theta = np.linspace(0, 2 * np.pi, circular_resolution, endpoint=False)
                cross_section_points = np.vstack((np.sin(theta), np.cos(theta))) * input_args.conductor_thickness
            else:
                cross_section_points = input_args.cross_sectional_points

            # Center the cross section around the origin [0, 0] for later rotation
            cross_section_points = cross_section_points - np.mean(cross_section_points, axis=1, keepdims=True)
            # Build a triangulation from the cross section
            cross_section_triangulation = Delaunay(cross_section_points.T)

            parameterized_mesh = coil_parts[part_ind].coil_mesh
            points_to_shift = coil_parts[part_ind].points_to_shift
            shift_array = coil_parts[part_ind].shift_array
            wire_path = coil_parts[part_ind].wire_path
            save_mesh = input_args.save_stl_flag
            output_directory = input_args.output_directory
            conductor_conductivity = input_args.specific_conductivity_conductor

            # Build a 2D mesh of the cross section by the corner points
            num_cross_section_points = cross_section_points.shape[1]
            cross_section_edges = np.vstack((np.arange(num_cross_section_points),
                                            np.roll(np.arange(num_cross_section_points), -1)))
            cross_section_triangulation = Delaunay(cross_section_points.T, qhull_options="QJ")
            cross_section_points = np.vstack((cross_section_triangulation.points.T, np.zeros(
                (1, cross_section_triangulation.points.shape[0]))))  # Embed the cross section in 3D
            is_interior_tri = cross_section_triangulation.find_simplex(cross_section_triangulation.points) >= 0
            cross_section_triangulation = cross_section_triangulation.simplices[is_interior_tri]

            # Calculate the area of the cross section
            cross_section_triangulation_3d = np.hstack(
                (cross_section_triangulation, np.zeros((cross_section_triangulation.shape[0], 1), dtype=np.int)))
            cross_section_area = 0

            for tri_ind in range(cross_section_triangulation_3d.shape[0]):
                P1 = cross_section_triangulation_3d[tri_ind, 0]
                P2 = cross_section_triangulation_3d[tri_ind, 1]
                P3 = cross_section_triangulation_3d[tri_ind, 2]
                cross_section_area += 0.5 * np.linalg.norm(np.cross(P2 - P1, P3 - P1))
            # Continue with Part 1
            """
            Conversion comments:
            In this Python code, NumPy functions are used to perform array operations, and the find_simplex method is 
            used to find the simplex that contains a point in the triangulation.
            """
            # Calculate the length of the coil
            wire_path.v_length = np.sum(np.linalg.norm(wire_path.v[:, 1:] - wire_path.v[:, :-1], axis=0))
            # Calculate the ohmic resistance
            ohmian_resistance = wire_path.v_length / (cross_section_area * conductor_conductivity)

            # Calculate a radius of the conductor cross section which is later
            # important to avoid intersection between angulated faces
            cross_section_center = np.sum(cross_section_points, axis=1) / cross_section_points.shape[1]
            cross_section_radius = np.max(np.linalg.norm(
                cross_section_points - cross_section_center[:, np.newaxis], axis=0))

            # Remove repeating entries
            diff_norm = np.linalg.norm(np.diff(wire_path.v, axis=1), axis=0)
            repeat_point_indices = np.where(diff_norm == 0)[0]
            wire_path.v = np.delete(wire_path.v, repeat_point_indices, axis=1)
            wire_path.uv = np.delete(wire_path.uv, repeat_point_indices, axis=1)

            # Open the track if it's not already opened
            point_inds_to_delete = np.where(np.linalg.norm(
                wire_path.v[:, -1][:, np.newaxis] - wire_path.v) < cross_section_radius / 2)[0]
            point_inds_to_delete = point_inds_to_delete[:round(len(point_inds_to_delete) / 2)]
            wire_path.v = np.delete(wire_path.v, point_inds_to_delete, axis=1)
            wire_path.uv = np.delete(wire_path.uv, point_inds_to_delete, axis=1)

            # Calculate the normal vectors along the wire track
            surface_normal_along_wire_path_v = np.zeros((3, wire_path.v.shape[1]))

            for point_ind in range(wire_path.v.shape[1]):
                node_ind_normals_target = planary_mesh_matlab_format.find_simplex(wire_path.uv[:, point_ind])

                if node_ind_normals_target == -1:  # Handle exceptions for strange output of pointLocation
                    if point_ind == 0:
                        surface_normal_along_wire_path_v[:,
                                                         point_ind] = surface_normal_along_wire_path_v[:, point_ind + 1]
                    else:
                        surface_normal_along_wire_path_v[:,
                                                         point_ind] = surface_normal_along_wire_path_v[:, point_ind - 1]
                else:
                    surface_normal_along_wire_path_v[:, point_ind] = parameterized_mesh.fn[node_ind_normals_target]

            # Continue with Part 2
            """
            Conversion comments:
            This Python code segment utilizes NumPy to perform vectorized operations, including calculations of 
            normalized vectors and cross products. The np.column_stack function is used to stack arrays as 
            columns. The loop constructs in the MATLAB code are translated into NumPy array operations for 
            efficiency.            
            """
            # Prepare the track direction
            path_directions = wire_path.v[:, 1:] - wire_path.v[:, :-1]  # edge vectors
            # add a repetition at the end for the last face
            path_directions = np.column_stack((path_directions, path_directions[:, -1]))
            path_directions = path_directions / np.linalg.norm(path_directions, axis=0)  # normalize

            path_directions[:, 1:-1] = (path_directions[:, :-2] + path_directions[:, 2:]) / \
                2  # average the vector between its predecessor and successor
            path_directions = path_directions / np.linalg.norm(path_directions, axis=0)  # normalize again

            # Sweep the surface along the path
            face_points = np.zeros((wire_path.v.shape[1], wire_path.v.shape[0],
                                   cross_section_points.shape[1]))  # initialize the face points
            all_node_points = np.zeros((wire_path.v.shape[1] * cross_section_points.shape[1], 3))
            run_ind = 0

            for point_ind in range(wire_path.v.shape[1]):
                e1 = path_directions[:, point_ind]
                e2 = surface_normal_along_wire_path_v[:, point_ind]
                e3 = np.cross(e1, e2) / np.linalg.norm(np.cross(e1, e2))
                e2 = np.cross(e1, e3) / np.linalg.norm(np.cross(e1, e3))

                # Rotate the face points to have the right orientation
                for aaaa in range(cross_section_points.shape[1]):
                    face_points[point_ind, :, aaaa] = np.dot(
                        np.column_stack((e2, e3, e1)), cross_section_points[:, aaaa])

                # Shift the oriented corner points to the position in the wire path
                for aaaa in range(cross_section_points.shape[1]):
                    face_points[point_ind, :, aaaa] += wire_path.v[:, point_ind]
                    all_node_points[run_ind, :] = face_points[point_ind, :, aaaa]
                    run_ind += 1

            # Continue with Part 3
            """
            Conversion comments:
            In this Python code fragment, NumPy arrays are used to handle indexing and vectorized operations. The 
            np.arange function is used to create arrays with a range of values. The modulo operation % is used to 
            ensure that the edge indices wrap around when reaching the last corner. The loop constructs in the MATLAB
            code are translated into NumPy array operations, including array slicing and modulo operations.
            """
            # Build the 3D mesh by sweeping the surface using only the outer edges

            # Building the shell triangles that form the surface of the swept body
            swept_surface_triangles = np.zeros(((face_points.shape[0] - 1) * face_points.shape[2] * 2, 3), dtype=int)
            swept_surface_vertices = all_node_points
            full_edge_inds = np.arange(1, face_points.shape[2] + 1)
            full_track_inds = np.arange(1, face_points.shape[0])
            num_corners = face_points.shape[2]

            run_ind = 1

            for track_ind in range(face_points.shape[0] - 1):

                for edge_ind in range(face_points.shape[2]):
                    node_a = track_ind * num_corners + full_edge_inds[edge_ind]
                    node_b = track_ind * num_corners + full_edge_inds[(edge_ind + 1) % num_corners]
                    node_c = full_track_inds[track_ind] * num_corners + full_edge_inds[edge_ind]
                    node_d = full_track_inds[track_ind] * num_corners + full_edge_inds[(edge_ind + 1) % num_corners]
                    tri_1 = [node_a, node_b, node_d]
                    tri_2 = [node_d, node_c, node_a]
                    swept_surface_triangles[run_ind - 1, :] = tri_1
                    swept_surface_triangles[run_ind, :] = tri_2
                    run_ind += 2

            # Part 4, final part
            """
            Conversion comments:
            In this translation, the Mesh data structure is used to represent the surface mesh. The os.path.join()
            function is used to construct file paths in a platform-independent manner. 
            The trimesh.exchange.export.export_mesh function is used to export the mesh as an STL file in ASCII format.
            Note that you should make sure that the necessary imports for the trimesh library are included at the 
            beginning of the code.
            """
            # Build the final triangles and close the surface
            run_ind = 1
            final_triangles = np.zeros((2 * num_corners, 3), dtype=int)
            left_end_nodes = np.arange((wire_path.v.shape[1] - 1) * num_corners, wire_path.v.shape[1] * num_corners)
            right_end_nodes = np.arange(0, num_corners)

            left_end_nodes = np.append(left_end_nodes, left_end_nodes[0])
            right_end_nodes = np.append(right_end_nodes, right_end_nodes[0])

            for edge_ind in range(num_corners):
                node_a = left_end_nodes[edge_ind]
                node_b = left_end_nodes[edge_ind + 1]
                node_c = right_end_nodes[edge_ind]
                node_d = right_end_nodes[edge_ind + 1]
                tri_1 = [node_a, node_b, node_d]
                tri_2 = [node_d, node_c, node_a]
                final_triangles[run_ind - 1, :] = tri_1
                final_triangles[run_ind, :] = tri_2
                run_ind += 2

            swept_faces = np.vstack((swept_surface_triangles, final_triangles))
            layout_surface_mesh = Mesh(vertices=swept_surface_vertices, faces=swept_faces)

            # Save the mesh as an .stl file
            if save_mesh:
                filename = input_args.field_shape_function.replace(
                    '*', '').replace('.', '').replace('^', '').replace(',', '')
                stl_file_path_layout = path.join(output_directory, f"swept_layout_part{part_ind}_{filename}.stl")
                stl_file_path_surface = path.join(output_directory, f"surface_part{part_ind}_{filename}.stl")

                trimesh.exchange.export.export_mesh(layout_surface_mesh.trimesh_obj,
                                                    stl_file_path_layout, file_type='stl_ascii')
                trimesh.exchange.export.export_mesh(curved_mesh_matlab_format,
                                                    stl_file_path_surface, file_type='stl_ascii')

            # Assign outputs
            coil_parts[part_ind].layout_surface_mesh = layout_surface_mesh
            coil_parts[part_ind].ohmian_resistance = ohmian_resistance

        return coil_parts


"""
Conversion comments:
Please note that in this Python code, the Delaunay function from scipy.spatial is used for triangulation
and the equivalent operations have been adapted accordingly. Also, the np namespace is used for various 
NumPy operations.
"""
