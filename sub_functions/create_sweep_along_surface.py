import numpy as np
from typing import List

from os import path
# Local functions
from sub_functions.data_structures import CoilPart, Mesh
from helpers.triangulation import Triangulate

# Logging
import logging

log = logging.getLogger(__name__)

# TODO: DEBUG Remove these
from helpers.visualisation import compare, visualize_faces
from sub_functions.data_structures import Shape2D, Shape3D

def create_sweep_along_surface(coil_parts: List[CoilPart], input_args, m_c_part = None) -> List[CoilPart]:
    """
    Create a volumetric coil body by surface sweep.

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures containing coil_mesh.
        input_args: Input arguments structure.

    Initialises the following properties of a CoilPart:
        - layout_surface_mesh
        - ohmian_resistance
        - wire_path.v_length

    Updates the following properties of a CoilPart:
        - wire_path.uv
        - wire_path.v
        
    Returns:
        List[CoilPart]: List of CoilPart structures with modified properties.
    """
    
    if not input_args.skip_sweep:

        save_mesh = input_args.save_stl_flag
        output_directory = input_args.output_directory
        conductor_conductivity = input_args.specific_conductivity_conductor

        for part_ind in range(len(coil_parts)):
            coil_part = coil_parts[part_ind]
            coil_mesh = coil_part.coil_mesh
            wire_path = coil_part.wire_path

            # DEBUG
            if m_c_part is not None:
                m_debug_out = m_c_part.create_sweep_along_surface
                m_connectivity = m_debug_out.ConnectivityList1
                m_connectivity_points = m_debug_out.ConnectivityPoints

            # Define the cross section of the conductor
            if np.all(input_args.cross_sectional_points == 0):
                circular_resolution = 10
                theta = np.linspace(0, 2 * np.pi, circular_resolution, endpoint=False)
                cross_section_points = np.vstack((np.sin(theta), np.cos(theta))) * input_args.conductor_thickness
            else:
                cross_section_points = input_args.cross_sectional_points

            # DEBUG
            if m_c_part is not None:
                assert(compare(cross_section_points, m_debug_out.cross_section_points1))

            # Center the cross section around the origin [0, 0] for later rotation
            cross_section_points = cross_section_points - np.mean(cross_section_points, axis=1, keepdims=True)

            # DEBUG
            if m_c_part is not None:
                assert(compare(cross_section_points, m_debug_out.cross_section_points2))

            # Build a triangulation from the cross section
            cross_section_points = cross_section_points[:, 0:-1]

            # DEBUG
            if m_c_part is not None:
                assert(compare(cross_section_points, m_debug_out.cross_section_points3))


            # Build a 2D mesh of the cross section by the corner points
            cross_section_points_2d = cross_section_points.T
            
            """
            zeros = np.zeros((cross_section_points_2d.shape[0], 1))
            cross_section_points_3d = np.concatenate((cross_section_points_2d, zeros), axis=1)

            num_cross_section_points = cross_section_points.shape[1]
            cross_section_edges = np.vstack((np.arange(num_cross_section_points),
                                            np.roll(np.arange(num_cross_section_points), -1))) # Python shape
            
            # DEBUG
            if m_c_part is not None:
                m_cross_section_edges = m_debug_out.cross_section_edges - 1
                assert(compare(cross_section_edges, m_cross_section_edges))

            # cross_section_triangulation = delaunayTriangulation(cross_section_points', cross_section_edges');
            cross_section_triangulation = Mesh(vertices=cross_section_points, faces=cross_section_edges)

            points1 = cross_section_triangulation.get_vertices() # Python shape
            points2 = np.zeros((points1.shape[1]))
            cross_section_points = np.vstack((points1, points2))  # Embed the cross section in 3D

            # DEBUG
            if m_c_part is not None:
                assert(compare(points1, m_debug_out.points1))
                assert(compare(points2, m_debug_out.points2))

            cross_section_faces = cross_section_triangulation.get_faces()
            cross_section_vertices = cross_section_triangulation.get_vertices()

            trimesh = cross_section_triangulation.trimesh_obj
            # facets_boundary

            # is_interior_tri = isInterior(cross_section_triangulation); # Boolean column vector. Each row indicates if the vertex is inside the shape.
            # Compute the Mesh consisting only of faces inside the cross-section
            #is_interior_tri = 1
            # inside_edges = cross_section_triangulation.ConnectivityList(is_interior_tri, :);
            #inside_faces = cross_section_faces[is_interior_tri]
            inside_faces = cross_section_faces
            """
            # TODO: Create a triangulation mesh from the vertices.
            # Trimesh can not do it.
            cross_section_triangulation = Triangulate(cross_section_points_2d)
            zeros = np.zeros((cross_section_points_2d.shape[0], 1))
            cross_section_points_3d = np.concatenate((cross_section_points_2d, zeros), axis=1)
            cross_section_triangulation_3d = Mesh(vertices=cross_section_points_3d, faces=cross_section_triangulation.get_triangles())

            # Calculate the area of the cross section
            # cross_section_triangulation = triangulation(inside_edges, cross_section_triangulation.Points);
            cross_section_area = 0

            vertices3d = cross_section_triangulation_3d.get_vertices()
            faces3d = cross_section_triangulation_3d.get_faces()

            for tri_ind, face in enumerate(faces3d):
                P1 = vertices3d[face[0]]
                P2 = vertices3d[face[1]]
                P3 = vertices3d[face[2]]
                cross_section_area += 0.5 * np.linalg.norm(np.cross(P2 - P1, P3 - P1))

            # DEBUG
            if m_c_part is not None:
                assert np.isclose(cross_section_area, m_debug_out.cross_section_area)

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

            # DEBUG
            if m_c_part is not None:
                assert np.isclose(cross_section_area, m_debug_out.cross_section_area)
                assert np.isclose(wire_path.v_length, m_debug_out.wire_path1.v_length)


            # Calculate a radius of the conductor cross section which is later
            # important to avoid intersection between angulated faces
            cross_section_center = np.sum(cross_section_points, axis=1) / cross_section_points.shape[1]
            cross_section_radius = np.max(np.linalg.norm(
                cross_section_points - cross_section_center[:, np.newaxis], axis=0))

            # DEBUG
            if m_c_part is not None:
                assert compare(cross_section_center, m_debug_out.cross_section_center[:2])
                assert compare(wire_path.v.T, m_debug_out.wire_path1.v.T) # , double_tolerance=0.03) # 
                assert cross_section_radius == m_debug_out.cross_section_radius


            # Remove repeating entries
            diff_norm = np.linalg.norm(np.diff(wire_path.v, axis=1), axis=0)
            repeat_point_indices = np.where(diff_norm == 0)[0]
            wire_path.v = np.delete(wire_path.v, repeat_point_indices, axis=1)
            wire_path.uv = np.delete(wire_path.uv, repeat_point_indices, axis=1)

            # DEBUG
            if m_c_part is not None:
                assert compare(wire_path.v.T, m_debug_out.wire_path2.v.T) # , double_tolerance=0.03) # 

            # Open the track if it's not already opened
            diff2_arr = wire_path.v[:, -1][:, np.newaxis] - wire_path.v
            diff2_norm = np.linalg.norm(diff2_arr, axis=0)
            arr2 =(diff2_norm < cross_section_radius / 2).astype(int)
            point_inds_to_delete = np.where(diff2_norm < cross_section_radius / 2)[0]

            # DEBUG
            if m_c_part is not None:
                assert compare(diff2_arr, m_debug_out.diff2_arr) # , double_tolerance=0.03)    # 
                assert compare(diff2_norm, m_debug_out.diff2_norm) # , double_tolerance=0.03)  # 
                assert compare(arr2, m_debug_out.point_inds_to_delete1) # M 1539 array of 0 or 1

            point_inds_to_delete = point_inds_to_delete[:round(len(point_inds_to_delete) / 2)]

            # DEBUG
            if m_c_part is not None:
                assert compare(point_inds_to_delete, m_debug_out.point_inds_to_delete2) # Y

            wire_path.v = np.delete(wire_path.v, point_inds_to_delete, axis=1)
            wire_path.uv = np.delete(wire_path.uv, point_inds_to_delete, axis=1)

            # Calculate the normal vectors along the wire track
            surface_normal_along_wire_path_v = np.zeros((3, wire_path.v.shape[1])) # MATLAB shape

            # planary_mesh_matlab_format = triangulation(parameterized_mesh.faces', parameterized_mesh.uv');
            wire_mesh2D = Mesh(vertices=coil_mesh.uv, faces=coil_mesh.get_faces())

            # DEBUG
            if m_c_part is not None:
                assert compare(wire_path.v.T, m_debug_out.wire_path3.v.T, double_tolerance=0.001) # Y
                assert compare(coil_mesh.uv, m_c_part.coil_mesh.uv.T) # ???
                assert compare(coil_mesh.get_faces(), m_c_part.coil_mesh.faces.T-1) # ???

                m_found_normals = m_debug_out.found_normals - 1
                m_found_points = m_debug_out.found_points.T
                a = 0
                b = 0
                zap = 3

            # TODO: There is something about the wire_mesh2D.get_face_index and pointLocation implementation that returns different results compared to MATLAB.
            for point_ind in range(wire_path.v.shape[1]):
                #log.debug(" point %d", point_ind)
                # node_ind_normals_target = pointLocation(planary_mesh_matlab_format, point_x, point_y);
                point = wire_path.uv[:, point_ind]
                node_ind_normals_target, possible_face_indices, faces_to_try = wire_mesh2D.get_face_index(point)

                # DEBUG
                if m_c_part is not None:
                    assert compare(point, m_found_points[point_ind])
                    if node_ind_normals_target != m_found_normals[point_ind]:
                        if a < zap:
                            log.debug(" faces don't match: %s, %s <> %s - %s", point_ind, node_ind_normals_target, m_found_normals[point_ind], possible_face_indices)
                            p_id = node_ind_normals_target
                            p_face_indices = coil_mesh.get_faces()[p_id]
                            p_vertices = coil_mesh.uv[p_face_indices]

                            m_id = m_found_normals[point_ind]
                            m_face_indices = (m_c_part.coil_mesh.faces.T-1)[m_id]
                            m_vertices = coil_mesh.uv[m_face_indices]

                            p_faces = [p_vertices]
                            m_faces = [m_vertices]

                            centres = point.reshape(1,2).T
                            
                            visualize_faces(p_faces, 800, f'images/19_get_face_index1_{point_ind}_p.png', centres)
                            visualize_faces(m_faces, 800, f'images/19_get_face_index1_{point_ind}_m.png', centres)

                            a += 1


                if node_ind_normals_target == -1:  # Handle exceptions for strange output of pointLocation

                    if m_c_part is not None:
                        if b < zap:
                            log.debug(" No face found: %d out of %d", point_ind, len(faces_to_try))
                            centres = point.reshape(1,2).T
                            p_contour = Shape2D()
                            x_uv = Shape2D()
                            face_vertices = []
                            for face in coil_mesh.uv[faces_to_try]:
                                face_vertices.append(face)
                            visualize_faces(face_vertices, 800, f'images/19_get_face_index2_{point_ind}_p.png', centres)
                            b += 1


                    if point_ind == 0:
                        surface_normal_along_wire_path_v[:,
                                                         point_ind] = surface_normal_along_wire_path_v[:, point_ind + 1]
                    else:
                        surface_normal_along_wire_path_v[:,
                                                         point_ind] = surface_normal_along_wire_path_v[:, point_ind - 1]
                else:
                    surface_normal_along_wire_path_v[:, point_ind] = coil_mesh.fn[node_ind_normals_target]


            # DEBUG
            if m_c_part is not None:
                assert compare(coil_mesh.fn, m_c_part.coil_mesh.fn)
                ## FAIL The face search above yields different faces... continuing.
                # assert compare(surface_normal_along_wire_path_v.T, m_debug_out.surface_normal_alonge_wire_path.v.T, double_tolerance=0.01)

            # DEBUG
            #log.warning(" Using MATLAB surface normals")
            #surface_normal_along_wire_path_v = m_debug_out.surface_normal_alonge_wire_path.v

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
                    arr1 = np.column_stack((e2, e3, e1))
                    arr2 = np.hstack((cross_section_points[:, aaaa], 0))
                    face_points[point_ind, :, aaaa] = np.dot(arr1, arr2)

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
            full_edge_inds = np.arange(0, face_points.shape[2])
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

                layout_surface_mesh.cleanup()
                layout_surface_mesh.export(stl_file_path_layout)
                coil_mesh.export(stl_file_path_surface)

                # DEBUG
                #layout_surface_mesh.display()
                m_swept_vertices=m_debug_out.swept_vertices
                m_swept_faces=m_debug_out.swept_faces-1

                if m_c_part is not None:
                    assert compare(swept_surface_vertices, m_swept_vertices, double_tolerance=0.003)
                    assert compare(swept_faces, m_swept_faces)
                #mat_mesh = Mesh(vertices=swept_vertices, faces=swept_faces)
                #mat_mesh.display()

            # Assign outputs
            coil_part.layout_surface_mesh = layout_surface_mesh
            coil_part.ohmian_resistance = ohmian_resistance

        return coil_parts


"""
Conversion comments:
Please note that in this Python code, the Delaunay function from scipy.spatial is used for triangulation
and the equivalent operations have been adapted accordingly. Also, the np namespace is used for various 
NumPy operations.
"""
