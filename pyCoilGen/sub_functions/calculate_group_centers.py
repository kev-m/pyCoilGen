import numpy as np
from typing import List

# Logging
import logging

# Local imports
from .data_structures import CoilPart, Shape3D, Mesh
from .find_segment_intersections import find_segment_intersections
from .check_mutual_loop_inclusion import check_mutual_loop_inclusion
from .uv_to_xyz import barycentric_to_cartesian

log = logging.getLogger(__name__)


def calculate_group_centers(coil_parts: List[CoilPart]) -> List[CoilPart]:
    """
    Calculate group centers for each coil part.

    Initialises the following properties of a CoilPart:
        - group_centers

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): A list of CoilPart structures, each containing a coil_mesh.

    Returns:
        coil_parts (List[CoilPart]): The updated list of CoilParts
    """

    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        part_mesh = coil_part.coil_mesh

        # Calculate the total center of the coil part
        total_center = np.mean(part_mesh.uv, axis=0)

        group_centers_2d = np.zeros((2, len(coil_part.groups)))
        total_group_center_uv = np.zeros((2, len(coil_part.groups)))
        total_group_center_v = np.zeros((3, len(coil_part.groups)))

        for group_ind in range(len(coil_part.groups)):
            coil_group = coil_part.groups[group_ind]
            point_sum_uv = np.empty((2, len(coil_group.loops)))
            point_sum_v = np.empty((3, len(coil_group.loops)))
            for loop_ind in range(len(coil_group.loops)):
                loop = coil_group.loops[loop_ind]
                point_sum_uv[:, loop_ind] = np.mean(loop.uv, axis=1)
                point_sum_v[:, loop_ind] = np.mean(loop.v, axis=1)

            total_group_center_uv[:, group_ind] = np.mean(point_sum_uv, axis=1)
            total_group_center_v[:, group_ind] = np.mean(point_sum_v, axis=1)
            inner_center = np.mean(coil_group.loops[-1].uv, axis=1)

            # Check if the total group center is within the most inner loop of the group
            mean1 = np.mean(coil_group.loops[-1].uv, axis=1, keepdims=True)
            inner_test_loop = (coil_group.loops[-1].uv - mean1) * 0.9 + mean1
            total_group_center_is_in = check_mutual_loop_inclusion(total_group_center_uv[:, group_ind], inner_test_loop)

            if total_group_center_is_in == 1:
                # total group center and inner center are within inner loop
                group_centers_2d[:, group_ind] = total_group_center_uv[:, group_ind]
            else:
                scale_ind = 1000
                cut_line_x = np.array([inner_center[0] + (total_center[0] - inner_center[0]) *
                                      scale_ind, inner_center[0] - (total_center[0] - inner_center[0]) * scale_ind])
                cut_line_y = np.array([inner_center[1] + (total_center[1] - inner_center[1]) *
                                      scale_ind, inner_center[1] - (total_center[1] - inner_center[1]) * scale_ind])
                intersection_points = find_segment_intersections(
                    coil_group.loops[-1].uv, np.array([cut_line_x, cut_line_y]))

                line_cut_inner_total_x = intersection_points[0].uv[0, :]
                line_cut_inner_total_y = intersection_points[0].uv[1, :]

                if line_cut_inner_total_x.size == 0:
                    group_centers_2d[:, group_ind] = inner_center
                else:
                    # Sort the cut points for their distance to the inner center
                    arr1 = np.array([line_cut_inner_total_x, line_cut_inner_total_y]) - total_center[:, None]
                    dist_to_inner_center = np.linalg.norm(arr1, axis=0)  # axis=0 == along rows. 0 = x-axis, 1 = y-axis
                    min_ind = np.argsort(dist_to_inner_center)
                    arr1 = line_cut_inner_total_x[min_ind]
                    arr2 = line_cut_inner_total_y[min_ind]
                    inner_cut_point = np.mean(np.array([arr1, arr2]), axis=1)
                    group_centers_2d[:, group_ind] = inner_cut_point

        # Set the centers, considering the possibility of non-mesh points
        group_centers_3d = np.zeros((3, group_centers_2d.shape[1]))

        planar_mesh = Mesh(faces=part_mesh.get_faces(), vertices=part_mesh.uv)
        curved_mesh = part_mesh.trimesh_obj  # Trimesh(faces=part_mesh.get_faces(), vertices=part_mesh.get_vertices())

        for rrrr in range(len(coil_part.groups)):
            # Set centers outside the 2D mesh in the center of the 3D volume
            point = [group_centers_2d[0, rrrr], group_centers_2d[1, rrrr]]
            target_triangle, bary_centric_coord = planar_mesh.get_face_index(
                point)  # get_target_triangle_def(point, planar_mesh)

            # TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
            if target_triangle != -1:
                vertices = curved_mesh.vertices[curved_mesh.faces[target_triangle]]
                group_centers_3d[:, rrrr] = barycentric_to_cartesian(bary_centric_coord, vertices)
            else:
                group_centers_3d[:, rrrr] = total_group_center_v[:, rrrr]

        # Set the group centers in the CoilPart structure
        coil_part.group_centers = Shape3D(uv=group_centers_2d, v=group_centers_3d)

    return coil_parts


"""
Please note that the find_segment_intersections, check_mutual_loop_inclusion, triangulation, pointLocation, and
barycentricToCartesian functions are not provided in the given Matlab code. You will need to implement these
functions in Python based on your specific requirements and data structures. Additionally, you might need to
adapt any additional class methods for the Mesh and related classes as mentioned in the comments.
"""
