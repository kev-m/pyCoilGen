import numpy as np

# Logging
import logging

from .data_structures import Shape3D
from pyCoilGen.helpers.common import nearest_approaches

log = logging.getLogger(__name__)


def open_loop_with_3d_sphere(curve_points_in: Shape3D, sphere_center: np.ndarray, sphere_diameter: float):
    """
    Opening a loop by overlapping it with a 3D sphere with a given radius and center position.

    NOTE: Uses MATLAB shape conventions

    Args:
        curve_points_in (CurvePoints): The input curve points.
        sphere_center (ndarray): (1,3) The center position of the sphere (3D coordinates).
        sphere_diameter (float): The diameter of the sphere.

    Returns:
        opened_loop (Shape3D), uv_cut (np.ndarray), cut_points (Shape3D): The opened loop, 2D contour of the cut shape, and cut points.
    """
    curve_points = curve_points_in.copy()  # Copy so that edits of curve_points do not affect source curve_points_in

    # Remove doubled points from the curve
    diff_array = curve_points.v[:, 1:] - curve_points.v[:, :-1]
    # Wrap the diff, like MATLAB does
    wrapped = curve_points.v[:, 0] - curve_points.v[:, -1]
    wrapped_array = [[wrapped[0]], [wrapped[1]], [wrapped[2]]]  # MATLAB shape
    diff_array = np.hstack((diff_array, wrapped_array))
    diff_array_norm = np.linalg.norm(diff_array, axis=0)
    indices_to_delete = np.where(abs(diff_array_norm) < 1e-10)

    if len(indices_to_delete[0]) > 0:
        curve_points.v = np.delete(curve_points.v, indices_to_delete, axis=1)
        curve_points.uv = np.delete(curve_points.uv, indices_to_delete, axis=1)
    curve_points.number_points = curve_points.v.shape[1]

    # Add a point within the curve which has the shortest distance to the sphere
    curve_points, near_points = add_nearest_ref_point_to_curve(curve_points, sphere_center)  # P: 3x57, M 3x58
    inside_sphere_ind = np.linalg.norm(curve_points.v - sphere_center, axis=0) < sphere_diameter / 2

    # Circshift the path so that the starting location is outside the sphere
    if np.any(inside_sphere_ind) and not np.all(inside_sphere_ind):
        min_ind = np.min(np.where(~inside_sphere_ind))
        curve_points.v = np.roll(curve_points.v, -min_ind, axis=1)
        curve_points.uv = np.roll(curve_points.uv, -min_ind, axis=1)
        inside_sphere_ind = np.linalg.norm(curve_points.v - sphere_center, axis=0) < sphere_diameter / 2
        if inside_sphere_ind[-1]:  # Shift again to avoid problems at the end of the curve
            curve_points.v = np.roll(curve_points.v, -1, axis=1)
            curve_points.uv = np.roll(curve_points.uv, -1, axis=1)
    else:
        raise ValueError("Opening of loop not possible, no overlap between cut sphere and loop")

    inside_sphere_ind = np.linalg.norm(curve_points.v - sphere_center, axis=0) < sphere_diameter / 2
    ind_diff_array = np.diff(inside_sphere_ind.astype(int))
    # In case of multiple cuts with the sphere, select the part of the curve which is closer to the sphere center
    if np.sum(np.abs(ind_diff_array)) > 2:
        parts_start = np.where(ind_diff_array == 1)[0] + 1
        parts_end = np.where(ind_diff_array == -1)[0]
        parts_avg_dist = np.zeros(len(parts_start))

        for part_ind in range(len(parts_start)):
            parts_avg_dist[part_ind] = np.mean(np.linalg.norm(
                curve_points.v[:, parts_start[part_ind]:parts_end[part_ind]] - sphere_center, axis=0))

        try:
            nearest_part = np.nanargmin(parts_avg_dist)
            inside_sphere_ind_unique = np.zeros(inside_sphere_ind.shape, dtype=bool)
            inside_sphere_ind_unique[parts_start[nearest_part]:parts_end[nearest_part] + 1] = True
        except ValueError as e:
            log.debug(" Caught ValueError exception, setting indices to False")
            inside_sphere_ind_unique = np.zeros(inside_sphere_ind.shape, dtype=bool)
    else:
        inside_sphere_ind_unique = inside_sphere_ind

    # Find the positions where the curve enters the sphere
    first_sphere_penetration_locations = np.where(np.abs(np.diff(inside_sphere_ind_unique)) == 1)[0]
    second_sphere_penetration_locations = first_sphere_penetration_locations + 1

    first_distances = np.linalg.norm(curve_points.v[:, first_sphere_penetration_locations] - sphere_center, axis=0)
    second_distances = np.linalg.norm(curve_points.v[:, second_sphere_penetration_locations] - sphere_center, axis=0)

    sphere_crossing_vecs = Shape3D(v=curve_points.v[:, second_sphere_penetration_locations] - curve_points.v[:, first_sphere_penetration_locations],
                                   uv=curve_points.uv[:, second_sphere_penetration_locations] - curve_points.uv[:, first_sphere_penetration_locations])

    # Calculate the penetration points by means of interpolation of weighted mean for the radial distance
    repeated_radii = np.ones(first_distances.shape) * sphere_diameter / 2

    # Open the loop; Check which parts of the curve are inside or outside the sphere
    try:
        shift_ind = np.min(np.where(inside_sphere_ind_unique == True)) * (-1)

        cut_points = Shape3D(v=curve_points.v[:, first_sphere_penetration_locations] + sphere_crossing_vecs.v * ((repeated_radii - first_distances) / (second_distances - first_distances)),
                             uv=curve_points.uv[:, first_sphere_penetration_locations] + sphere_crossing_vecs.uv * ((repeated_radii - first_distances) / (second_distances - first_distances)))

        curve_points.v = np.roll(curve_points.v, shift_ind, axis=1)
        curve_points.uv = np.roll(curve_points.uv, shift_ind, axis=1)

        inside_sphere_ind_unique = np.roll(inside_sphere_ind_unique, shift_ind)

        curve_points.v = curve_points.v[:, ~inside_sphere_ind_unique]
        curve_points.uv = curve_points.uv[:, ~inside_sphere_ind_unique]

        # Build the "opened" loop with the cut_points as open ends
        # Remove curve points which are still inside the sphere
        finished_loop_case1 = Shape3D(v=np.hstack((cut_points.v[:, [0]], curve_points.v, cut_points.v[:, [-1]])),
                                      uv=np.hstack((cut_points.uv[:, [0]], curve_points.uv, cut_points.uv[:, [-1]])))
        finished_loop_case2 = Shape3D(v=np.hstack((cut_points.v[:, [-1]], curve_points.v, cut_points.v[:, [0]])),
                                      uv=np.hstack((cut_points.uv[:, [-1]], curve_points.uv, cut_points.uv[:, [0]])))

        mean_dist_1 = np.sum(np.linalg.norm(finished_loop_case1.v[:, 1:] - finished_loop_case1.v[:, :-1], axis=0))
        mean_dist_2 = np.sum(np.linalg.norm(finished_loop_case2.v[:, 1:] - finished_loop_case2.v[:, :-1], axis=0))

        if mean_dist_1 < mean_dist_2:
            opened_loop = Shape3D(v=finished_loop_case1.v, uv=finished_loop_case1.uv)
        else:
            opened_loop = Shape3D(v=finished_loop_case2.v, uv=finished_loop_case2.uv)
    except ValueError as e:
        log.debug(" Caught ValueError exception, setting cut_points to empty")
        opened_loop = Shape3D(v=curve_points.v, uv=curve_points.uv)
        cut_points = Shape3D(uv=np.array([]), v=np.array([]))

    # Generate the 2d contour of the cut shape for later plotting
    radius_2d = np.linalg.norm(opened_loop.uv[:, [0]] - opened_loop.uv[:, [-1]]) / 2
    sin_cos_arr = np.array([
        np.sin([i / (50 / (2 * np.pi)) for i in range(51)]),
        np.cos([i / (50 / (2 * np.pi)) for i in range(51)])
    ])

    uv_cut = np.array(sin_cos_arr) * radius_2d + (opened_loop.uv[:, [0]] + opened_loop.uv[:, [-1]]) / 2
    return opened_loop, uv_cut, cut_points


# NOTE: curve_track_in and target_point are assumed to use MATLAB shape
def add_nearest_ref_point_to_curve(curve_track_in: Shape3D, target_point: np.ndarray):
    """
    Calculate the mutual nearest positions and segment indices between two loops.

    NOTE: Uses MATLAB shape conventions

    Args:
        curve_track_in (ContourLine): The input curve points.
        target_point (ndarray): The target point (3D coordinates) (1,3).

    Returns:
       curve_track_out, near_points (Shape3D, Shape2D): The updated curve track and the nearest points.
    """
    # Create a copy of the input contour line to avoid modifying the original
    curve_track = curve_track_in.copy()

    # Check if the contour line is open, if so, close it by duplicating the first point
    if (curve_track.v[:, 0] != curve_track.v[:, -1]).any():
        curve_track.v = np.column_stack((curve_track.v, curve_track.v[:, 0]))
        curve_track.uv = np.column_stack((curve_track.uv, curve_track.uv[:, 0]))

    # Extract segment start and end points
    seg_starts = Shape3D(v=curve_track.v[:, :-1], uv=curve_track.uv[:, :-1])
    seg_ends = Shape3D(v=curve_track.v[:, 1:], uv=curve_track.uv[:, 1:])

    # Calculate the parameter 't' to find the nearest point on each segment to the target point
    t, vec_segs = nearest_approaches(target_point, seg_starts.v, seg_ends.v)
    t[t < 0] = 0
    t[t > 1] = 1

    # Calculate all near points on each segment
    all_near_points_v = seg_starts.v + vec_segs * np.tile(t, (3, 1))
    all_near_points_uv = seg_starts.uv + (seg_ends.uv - seg_starts.uv) * np.tile(t, (2, 1))

    # Calculate distances from all_near_points to the target point
    all_dists = np.linalg.norm(all_near_points_v - target_point, axis=0)

    # Find the nearest point (minimum distance) and its index
    min_ind_seq = np.argmin(all_dists)

    # Extract the nearest point
    near_points = Shape3D(v=all_near_points_v[:, min_ind_seq], uv=all_near_points_uv[:, min_ind_seq])

    # Add the nearest point within the curve
    curve_track_out_v = np.column_stack((curve_track_in.v[:, :min_ind_seq+1], near_points.v))
    curve_track_out_uv = np.column_stack((curve_track_in.uv[:, :min_ind_seq+1], near_points.uv))

    # Check if the nearest point is not the last point in the contour line
    if min_ind_seq != curve_track_in.v.shape[1] - 1:  # -1 because of MATLAB
        curve_track_out_v = np.column_stack((curve_track_out_v, curve_track_in.v[:, min_ind_seq + 1:]))
        curve_track_out_uv = np.column_stack((curve_track_out_uv, curve_track_in.uv[:, min_ind_seq + 1:]))

    curve_track_out = Shape3D(v=curve_track_out_v, uv=curve_track_out_uv)

    return curve_track_out, near_points
