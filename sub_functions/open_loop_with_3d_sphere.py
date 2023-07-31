import numpy as np

from sub_functions.data_structures import ContourLine


def open_loop_with_3d_sphere(curve_points_in: ContourLine, sphere_center: np.ndarray, sphere_diameter: float):
    """
    Opening a loop by overlapping it with a 3D sphere with a given radius and center position.

    Args:
        curve_points_in (CurvePoints): The input curve points.
        sphere_center (ndarray): The center position of the sphere (3D coordinates).
        sphere_diameter (float): The diameter of the sphere.

    Returns:
        CurvePoints, ndarray, CurvePoints: The opened loop, 2D contour of the cut shape, and cut points.
    """

    # Remove doubled points from the curve
    points_to_delete = np.linalg.norm(curve_points_in.v[:, 1:] - curve_points_in.v[:, :-1], axis=0) < 1e-10
    curve_points_in.v = curve_points_in.v[:, ~points_to_delete]
    curve_points_in.uv = curve_points_in.uv[:, ~points_to_delete]
    curve_points_in.number_points = curve_points_in.v.shape[1]

    # Add a point within the curve which has the shortest distance to the sphere
    curve_points, _ = add_nearest_ref_point_to_curve(curve_points_in, sphere_center)

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

    # In case of multiple cuts with the sphere, select the part of the curve which is closer to the sphere center
    if np.sum(np.abs(np.diff(inside_sphere_ind))) > 2:
        parts_start = np.where(np.diff(inside_sphere_ind) == 1)[0] + 1
        parts_end = np.where(np.diff(inside_sphere_ind) == -1)[0]
        parts_avg_dist = np.zeros(len(parts_start))

        for part_ind in range(len(parts_start)):
            parts_avg_dist[part_ind] = np.mean(np.linalg.norm(
                curve_points.v[:, parts_start[part_ind]:parts_end[part_ind]] - sphere_center, axis=0))

        nearest_part = np.argmin(parts_avg_dist)
        inside_sphere_ind_unique = np.zeros(inside_sphere_ind.shape, dtype=bool)
        inside_sphere_ind_unique[parts_start[nearest_part]:parts_end[nearest_part] + 1] = True
    else:
        inside_sphere_ind_unique = inside_sphere_ind

    # Find the positions where the curve enters the sphere
    first_sphere_penetration_locations = np.where(np.abs(np.diff(inside_sphere_ind_unique)) == 1)[0]
    second_sphere_penetration_locations = first_sphere_penetration_locations + 1

    first_distances = np.linalg.norm(curve_points.v[:, first_sphere_penetration_locations] - sphere_center, axis=0)
    second_distances = np.linalg.norm(curve_points.v[:, second_sphere_penetration_locations] - sphere_center, axis=0)

    sphrere_crossing_vecs = {'v': curve_points.v[:, second_sphere_penetration_locations] - curve_points.v[:, first_sphere_penetration_locations],
                             'uv': curve_points.uv[:, second_sphere_penetration_locations] - curve_points.uv[:, first_sphere_penetration_locations]}

    # Calculate the penetration points by means of interpolation of weighted mean for the radial distance
    repeated_radia = np.ones(first_distances.shape) * sphere_diameter / 2
    cut_points = {'v': curve_points.v[:, first_sphere_penetration_locations] + sphrere_crossing_vecs['v'] * ((repeated_radia - first_distances) / (second_distances - first_distances)),
                  'uv': curve_points.uv[:, first_sphere_penetration_locations] + sphrere_crossing_vecs['uv'] * ((repeated_radia - first_distances) / (second_distances - first_distances))}

    # Open the loop; Check which parts of the curve are inside or outside the sphere
    shift_ind = (np.min(np.where(inside_sphere_ind_unique == 1)) - 1) * (-1)
    curve_points.v = np.roll(curve_points.v, shift_ind, axis=1)
    curve_points.uv = np.roll(curve_points.uv, shift_ind, axis=1)
    inside_sphere_ind_unique = np.roll(inside_sphere_ind_unique, shift_ind)
    curve_points.v = curve_points.v[:, ~inside_sphere_ind_unique]
    curve_points.uv = curve_points.uv[:, ~inside_sphere_ind_unique]

    # Build the "opened" loop with the cut_points as open ends
    # Remove curve points which are still inside the sphere
    finished_loop_case1 = {'v': np.hstack((cut_points['v'][:, [0]], curve_points.v, cut_points['v'][:, [-1]])),
                           'uv': np.hstack((cut_points['uv'][:, [0]], curve_points.uv, cut_points['uv'][:, [-1]]))}
    finished_loop_case2 = {'v': np.hstack((cut_points['v'][:, [-1]], curve_points.v, cut_points['v'][:, [0]])),
                           'uv': np.hstack((cut_points['uv'][:, [-1]], curve_points.uv, cut_points['uv'][:, [0]]))}

    mean_dist_1 = np.sum(np.linalg.norm(finished_loop_case1['v'][:, 1:] - finished_loop_case1['v'][:, :-1], axis=0))
    mean_dist_2 = np.sum(np.linalg.norm(finished_loop_case2['v'][:, 1:] - finished_loop_case2['v'][:, :-1], axis=0))

    if mean_dist_1 < mean_dist_2:
        opened_loop = {'v': finished_loop_case1['v'], 'uv': finished_loop_case1['uv']}
    else:
        opened_loop = {'v': finished_loop_case2['v'], 'uv': finished_loop_case2['uv']}

    # Generate the 2d contour of the cut shape for later plotting
    radius_2d = np.linalg.norm(opened_loop['uv'][:, [0]] - opened_loop['uv'][:, [-1]]) / 2
    sin_cos_arr = [
        np.sin([i / (50 / (2 * np.pi)) for i in range(51)]),
        np.cos([i / (50 / (2 * np.pi)) for i in range(51)])
    ]
    uv_cut = np.array(sin_cos_arr) * radius_2d + (opened_loop['uv'][:, [0]] + opened_loop['uv'][:, [-1]]) / 2

    return opened_loop, uv_cut, cut_points


def add_nearest_ref_point_to_curve(curve_track_in, target_point):
    """
    Calculate the mutual nearest positions and segment indices between two loops.

    Args:
        curve_track_in (CurvePoints): The input curve points.
        target_point (ndarray): The target point (3D coordinates).

    Returns:
        CurvePoints, ndarray: The updated curve track and the nearest points.
    """

    curve_track = curve_track_in

    if not np.allclose(curve_track.v[:, 0], curve_track.v[:, -1]):
        curve_track.v = np.hstack((curve_track.v, curve_track.v[:, [0]]))
        curve_track.uv = np.hstack((curve_track.uv, curve_track.uv[:, [0]]))

    seg_starts = {'v': curve_track.v[:, :-1], 'uv': curve_track.uv[:, :-1]}
    seg_ends = {'v': curve_track.v[:, 1:], 'uv': curve_track.uv[:, 1:]}

    t = np.sum((target_point - seg_starts['v']) * (seg_ends['v'] - seg_starts['v']),
               axis=0) / np.sum((seg_ends['v'] - seg_starts['v'])**2, axis=0)
    t = np.clip(t, 0, 1)
    all_near_points = {'v': seg_starts['v'] + (seg_ends['v'] - seg_starts['v']) * t,
                       'uv': seg_starts['uv'] + (seg_ends['uv'] - seg_starts['uv']) * t}
    all_dists = np.linalg.norm(all_near_points['v'] - target_point, axis=0)
    min_ind_seq = np.argmin(all_dists)
    near_points = {'v': all_near_points['v'][:, [min_ind_seq]],
                   'uv': all_near_points['uv'][:, [min_ind_seq]]}

    curve_track_out = {'v': np.hstack((curve_track_in.v[:, :min_ind_seq], near_points['v'])),
                       'uv': np.hstack((curve_track_in.uv[:, :min_ind_seq], near_points['uv']))}

    if min_ind_seq != curve_track_in.v.shape[1] - 1:
        curve_track_out['v'] = np.hstack((curve_track_out['v'], curve_track_in.v[:, min_ind_seq + 1:]))
        curve_track_out['uv'] = np.hstack((curve_track_out['uv'], curve_track_in.uv[:, min_ind_seq + 1:]))

    return curve_track_out, near_points
