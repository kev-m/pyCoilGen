from helpers.visualisation import compare
import numpy as np

# Logging
import logging

from sub_functions.data_structures import Shape3D
from helpers.common import nearest_approaches

log = logging.getLogger(__name__)

# TODO: Remove this debugging helper
from helpers.visualisation import compare

def open_loop_with_3d_sphere(curve_points_in: Shape3D, sphere_center: np.ndarray, sphere_diameter: float, debug_data=None):
    """
    Opening a loop by overlapping it with a 3D sphere with a given radius and center position.

    NOTE: Uses MATLAB shape conventions

    Args:
        curve_points_in (CurvePoints): The input curve points.
        sphere_center (ndarray): The center position of the sphere (3D coordinates).
        sphere_diameter (float): The diameter of the sphere.

    Returns:
        CurvePoints, ndarray, CurvePoints: The opened loop, 2D contour of the cut shape, and cut points.
    """

    # Remove doubled points from the curve
    indices_to_delete = np.where(np.linalg.norm(curve_points_in.v[:, 1:] - curve_points_in.v[:, :-1], axis=0) < 1e-10)
    # Always remove last point
    curve_points_in.v = curve_points_in.v[:, :-1]
    curve_points_in.uv = curve_points_in.uv[:, :-1]

    np.delete(curve_points_in.v, indices_to_delete)
    np.delete(curve_points_in.uv, indices_to_delete)
    curve_points_in.number_points = curve_points_in.v.shape[1]

    # Add a point within the curve which has the shortest distance to the sphere
    curve_points, _ = add_nearest_ref_point_to_curve(curve_points_in, sphere_center)

    if debug_data is not None:
        log.debug(" 1  v: %s", compare(curve_points.v, debug_data['curve_points1'].v, double_tolerance=0.01))
        log.debug(" 1 uv: %s", compare(curve_points.uv, debug_data['curve_points1'].uv, double_tolerance=0.01))

    inside_sphere_ind = np.linalg.norm(curve_points.v - sphere_center, axis=0) < sphere_diameter / 2

    # Circshift the path so that the starting location is outside the sphere
    if np.any(inside_sphere_ind) and not np.all(inside_sphere_ind):
        min_ind = np.min(np.where(~inside_sphere_ind)) - 1
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

        nearest_part = np.nanargmin(parts_avg_dist)
        inside_sphere_ind_unique = np.zeros(inside_sphere_ind.shape, dtype=bool)
        inside_sphere_ind_unique[parts_start[nearest_part]:parts_end[nearest_part] + 1] = True
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
    cut_points = Shape3D(v=curve_points.v[:, first_sphere_penetration_locations] + sphere_crossing_vecs.v * ((repeated_radii - first_distances) / (second_distances - first_distances)),
                         uv=curve_points.uv[:, first_sphere_penetration_locations] + sphere_crossing_vecs.uv * ((repeated_radii - first_distances) / (second_distances - first_distances)))

    # Open the loop; Check which parts of the curve are inside or outside the sphere
    shift_ind = (np.min(np.where(inside_sphere_ind_unique == 1)) - 1) * (-1)
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

    # Generate the 2d contour of the cut shape for later plotting
    radius_2d = np.linalg.norm(opened_loop.uv[:, [0]] - opened_loop.uv[:, [-1]]) / 2
    sin_cos_arr = [
        np.sin([i / (50 / (2 * np.pi)) for i in range(51)]),
        np.cos([i / (50 / (2 * np.pi)) for i in range(51)])
    ]
    uv_cut = np.array(sin_cos_arr) * radius_2d + (opened_loop.uv[:, [0]] + opened_loop.uv[:, [-1]]) / 2

    return opened_loop, uv_cut, cut_points


# NOTE: curve_track_in and target_point are assumed to use MATLAB shape
# TODO: Remove debug_data
def add_nearest_ref_point_to_curve(curve_track_in: Shape3D, target_point: np.ndarray, debug_data=None):
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

    # debug_data.curve_track = curve_track;
    if debug_data is not None:
        log.debug(" 1  v: %s", compare(curve_track.v, debug_data['curve_track'].v, double_tolerance=0.001))
        log.debug(" 1 uv: %s", compare(curve_track.uv, debug_data['curve_track'].uv, double_tolerance=0.001))

    # Extract segment start and end points
    seg_starts = Shape3D(v=curve_track.v[:, :-1], uv=curve_track.uv[:, :-1])
    seg_ends = Shape3D(v=curve_track.v[:, 1:], uv=curve_track.uv[:, 1:])

    if debug_data is not None:
        log.debug(" 2 seg_starts.v: %s", compare(seg_starts.v, debug_data['seg_starts'].v, double_tolerance=0.001))
        log.debug(" 2 seg_starts.uv: %s", compare(seg_starts.uv, debug_data['seg_starts'].uv, double_tolerance=0.001))
        log.debug(" 3 seg_ends.v: %s", compare(seg_ends.v, debug_data['seg_ends'].v, double_tolerance=0.001))
        log.debug(" 3 seg_ends.uv: %s", compare(seg_ends.uv, debug_data['seg_ends'].uv, double_tolerance=0.001))

    # Calculate the parameter 't' to find the nearest point on each segment to the target point
    t, vec_segs = nearest_approaches(target_point, seg_starts.v, seg_ends.v)
    if debug_data is not None:
        log.debug(" 4 t1: %s", compare(t, debug_data['t1'], double_tolerance=0.001))
    t[t < 0] = 0
    if debug_data is not None:
        log.debug(" 4 t2: %s", compare(t, debug_data['t2'], double_tolerance=0.001))
    t[t > 1] = 1
    if debug_data is not None:
        log.debug(" 4 t3: %s", compare(t, debug_data['t3'], double_tolerance=0.001))

    # Calculate all near points on each segment
    all_near_points_v = seg_starts.v + vec_segs * np.tile(t, (3, 1))
    all_near_points_uv = seg_starts.uv + (seg_ends.uv - seg_starts.uv) * np.tile(t, (2, 1))

    if debug_data is not None:
        log.debug(" 5 all_near_points_v: %s", compare(all_near_points_v, debug_data['all_near_points'].v, double_tolerance=0.001))
        log.debug(" 5 all_near_points_uv: %s", compare(all_near_points_uv, debug_data['all_near_points'].uv, double_tolerance=0.001))

    # Calculate distances from all_near_points to the target point
    all_dists = np.linalg.norm(all_near_points_v - target_point, axis=0)
    if debug_data is not None:
        log.debug(" 6 all_dists: %s", compare(all_dists, debug_data['all_dists'], double_tolerance=0.001))

    # Find the nearest point (minimum distance) and its index
    min_ind_seq = np.argmin(all_dists)
    if debug_data is not None:
        log.debug(" 7 min_ind_seq: %s", min_ind_seq == debug_data['min_ind_seq']-1) # -1 because MATLAB is 1-based

    # Extract the nearest point
    near_points = Shape3D(v=all_near_points_v[:, min_ind_seq], uv=all_near_points_uv[:, min_ind_seq])

    if debug_data is not None:
        log.debug(" 8 near_points: %s", compare(near_points.v, debug_data['near_points'].v, double_tolerance=0.001))
        log.debug(" 8 near_points: %s", compare(near_points.uv, debug_data['near_points'].uv, double_tolerance=0.001))

    # Add the nearest point within the curve
    curve_track_out_v = np.column_stack((curve_track_in.v[:, :min_ind_seq+1], near_points.v))
    curve_track_out_uv = np.column_stack((curve_track_in.uv[:, :min_ind_seq+1], near_points.uv))

    # Check if the nearest point is not the last point in the contour line
    if min_ind_seq != curve_track_in.v.shape[1] - 1:
        #  curve_track_out.v = [curve_track_out.v curve_track_in.v(:, min_ind_seq + 1:end)];
        curve_track_out_v = np.column_stack((curve_track_out_v, curve_track_in.v[:, min_ind_seq + 1:]))
        curve_track_out_uv = np.column_stack((curve_track_out_uv, curve_track_in.uv[:, min_ind_seq + 1:]))

    curve_track_out = Shape3D(v=curve_track_out_v, uv=curve_track_out_uv)

    if debug_data is not None:
        log.debug(" 9 curve_track_out.v: %s", compare(curve_track_out.v, debug_data['curve_track_out'].v, double_tolerance=0.001))
        log.debug(" 9 curve_track_out.uv: %s", compare(curve_track_out.uv, debug_data['curve_track_out'].uv, double_tolerance=0.001))

    return curve_track_out, near_points


"""
    curve_track_out = Shape3D(v=np.hstack((curve_track_in.v[:, :min_ind_seq], near_points.v)),
                              uv=np.hstack((curve_track_in.uv[:, :min_ind_seq], near_points.uv)))

    if min_ind_seq != curve_track_in.v.shape[1] - 1:
        curve_track_out.v = np.hstack((curve_track_out.v, curve_track_in.v[:, min_ind_seq + 1:]))
        curve_track_out.uv = np.hstack((curve_track_out.uv, curve_track_in.uv[:, min_ind_seq + 1:]))

    return curve_track_out, near_points
"""
