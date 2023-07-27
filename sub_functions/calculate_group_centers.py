import numpy as np
from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart
from sub_functions.find_segment_intersections import find_segment_intersections
from sub_functions.check_mutual_loop_inclusion import check_mutual_loop_inclusion

log = logging.getLogger(__name__)


def calculate_group_centers(coil_parts: List[CoilPart]):
    """
    Calculate group centers for each coil part.

    Parameters:
        coil_parts (List[CoilPart]): A list of CoilPart structures, each containing a coil_mesh.

    Returns:
        None. The 'group_centers' attribute is added to each CoilPart in the input list.

    Example:
        # Create a list of CoilPart structures with coil_mesh information
        coil_parts = [CoilPart(coil_mesh=mesh1), CoilPart(coil_mesh=mesh2), ...]

        # Calculate group centers for each CoilPart
        calculate_group_centers(coil_parts)
    """

    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        part_mesh = coil_part.coil_mesh
        part_vertices = part_mesh.get_vertices()  # Get the vertices for the coil part.
        part_faces = part_mesh.get_faces()

        # Calculate the total center of the coil part
        total_center = np.mean(part_vertices, axis=1)

        group_centers_2d = np.zeros((2, len(coil_part.groups)))
        total_group_center_uv = np.zeros((2, len(coil_part.groups)))
        total_group_center_v = np.zeros((3, len(coil_part.groups)))

        for group_ind in range(len(coil_part.groups)):
            coil_group = coil_part.groups[group_ind]
            # M: point_sum_uv	2x387 double	2x387	double
            point_sum_uv = np.empty((2, len(coil_group.loops)))
            # M: point_sum_v	3x387 double	3x387	double
            point_sum_v = np.empty((3, len(coil_group.loops)))
            for loop_ind in range(len(coil_group.loops)):
                loop = coil_group.loops[loop_ind]
                point_sum_uv[:,loop_ind] = np.mean(loop.uv, axis=1)
                point_sum_v[:,loop_ind] = np.mean(loop.v, axis=1)

            # M: ans =
            #    -1.1346
            #     0.0226
            total_group_center_uv[:, group_ind] = np.mean(point_sum_uv, axis=1)
            # M: ans =
            #    -0.0026
            #     0.3506
            #    -0.3617
            total_group_center_v[:, group_ind] = np.mean(point_sum_v, axis=1)
            # M: ans =
            #    -1.5743
            #     0.0161
            inner_center = np.mean(coil_group.loops[-1].uv, axis=1)

            # Check if the total group center is within the most inner loop of the group
            mean1 = np.mean(coil_group.loops[-1].uv, axis=1, keepdims =True)
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

                # 'list' object has no attribute 'uv'
                log.debug(" -- here -- ")
                line_cut_inner_total_x = intersection_points.uv[0, :]
                line_cut_inner_total_y = intersection_points.uv[1, :]

                if line_cut_inner_total_x.size == 0:
                    group_centers_2d[:, group_ind] = inner_center
                else:
                    # Sort the cut points for their distance to the inner center
                    dist_to_inner_center = np.linalg.norm(
                        np.array([line_cut_inner_total_x, line_cut_inner_total_y]) - total_center[:, None], axis=0)
                    min_ind = np.argsort(dist_to_inner_center)
                    inner_cut_point = np.mean(
                        np.array([line_cut_inner_total_x[min_ind[0]], line_cut_inner_total_y[min_ind[0]]]), axis=1)
                    group_centers_2d[:, group_ind] = inner_cut_point

        # M: group_centers_2d	[-1.5890,0.9301,1.5624,-0.9478;0.0162,-0.0045,0.0646,-0.0042]	2x4	double
        #    -1.5890    0.9301    1.5624   -0.9478
        #     0.0162   -0.0045    0.0646   -0.0042
        # Set the centers, considering the possibility of non-mesh points
        group_centers_3d = np.zeros((3, group_centers_2d.shape[1]))

        planar_mesh = triangulation(coil_part.coil_mesh.faces.T, coil_part.coil_mesh.uv.T)
        curved_mesh = triangulation(coil_part.coil_mesh.faces.T, coil_part.coil_mesh.vertices)

        for rrrr in range(len(coil_part.groups)):
            # Set centers outside the 2D mesh in the center of the 3D volume
            target_triangle, bary_centric_coord = pointLocation(
                planar_mesh, group_centers_2d[0, rrrr], group_centers_2d[1, rrrr])

            if not np.isnan(target_triangle):
                group_centers_3d[:, rrrr] = barycentricToCartesian(curved_mesh, target_triangle, bary_centric_coord)
            else:
                group_centers_3d[:, rrrr] = total_group_center_v[:, rrrr]

        # Set the group centers in the CoilPart structure
        coil_parts[part_ind].group_centers = {'uv': group_centers_2d, 'v': group_centers_3d}


"""
Please note that the find_segment_intersections, check_mutual_loop_inclusion, triangulation, pointLocation, and
barycentricToCartesian functions are not provided in the given Matlab code. You will need to implement these
functions in Python based on your specific requirements and data structures. Additionally, you might need to
adapt any additional class methods for the Mesh and related classes as mentioned in the comments.
"""