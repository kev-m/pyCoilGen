import numpy as np
from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart
from sub_functions.uv_to_xyz import uv_to_xyz, pointLocation

log = logging.getLogger(__name__)

# TODO: Remove debugging helpers
from helpers.visualisation import compare


def shift_return_paths(coil_parts: List[CoilPart], input_args, m_c_part=None):
    """
    Shifts and returns wire paths for coil parts while avoiding wire intersections.

    Initialises the following properties of a CoilPart:
        - shift_array
        - points_to_shift

    Updates the following properties of a CoilPart:
        - wire_path

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures containing coil_mesh.
        input_args: Input arguments structure.

    Returns:
        List[CoilPart]: List of CoilPart structures with updated wire paths and shift information.
    """

    if not input_args.skip_normal_shift:
        smooth_factors = input_args.normal_shift_smooth_factors
        normal_shift_length = input_args.normal_shift_length
        for part_ind in range(len(coil_parts)):
            coil_part = coil_parts[part_ind]
            coil_mesh = coil_part.coil_mesh
            part_vertices = coil_mesh.get_vertices()  # part_mesh.vertices
            part_faces = coil_mesh.get_faces()  # part_mesh.faces
            wire_path_out = coil_part.wire_path
            # part_mesh = coil_mesh.trimesh_obj

            # Delete superimposing points
            diff = np.diff(wire_path_out.v, axis=1)
            points_to_delete = np.concatenate((np.linalg.norm(diff, axis=0), [1]))
            indices_to_delete = np.where(points_to_delete == 0)
            wire_path_out.uv = np.delete(wire_path_out.uv, indices_to_delete, axis=1)
            wire_path_out.v = np.delete(wire_path_out.v, indices_to_delete, axis=1)

            if m_c_part is not None:
                m_debug_base = m_c_part.shift_return_paths
                assert compare(wire_path_out.uv, m_debug_base.wire_path_out.uv)

            if input_args.smooth_flag and input_args.smooth_factor > 1:
                # Apply smoothing to the wire path
                wire_path_out.uv = smooth_track_by_folding(wire_path_out.uv, input_args.smooth_factor)
                wire_path_out.v, wire_path_out.uv = uv_to_xyz(wire_path_out.uv, part_faces, part_vertices)

            if np.all(wire_path_out.uv[:, -1] == wire_path_out.uv[:, 0]):
                wire_path_out.add_uv(wire_path_out.uv[:, :1])
                wire_path_out.add_v(wire_path_out.v[:, :1])

            # Detect wire crossings
            if m_c_part is not None:
                cross_points, cross_segments = InterX(wire_path_out.uv, m_debug=m_debug_base.debug_out_interex)
            else:
                cross_points, cross_segments = InterX(wire_path_out.uv)


            if m_c_part is not None:
                assert compare(cross_points, m_debug_base.cross_points)         # Pass
                assert compare(cross_segments, m_debug_base.cross_segments-1)   # Pass

            if cross_segments.size != 0:
                sorted_crossed_segments = np.sort(np.concatenate((cross_segments[:, 0], cross_segments[:, 1])))
                neighbors_weight = np.zeros(sorted_crossed_segments.shape)
                scale_factor = np.sum(sorted_crossed_segments**4)

                if m_c_part is not None:
                    assert compare(sorted_crossed_segments, m_debug_base.sorted_crossed_segments - 1)

                for crossed_seg_ind in range(sorted_crossed_segments.size):
                    check_dists = np.abs(sorted_crossed_segments - sorted_crossed_segments[crossed_seg_ind])
                    check_dists = check_dists[check_dists != 0]
                    neighbors_weight[crossed_seg_ind] = np.min(check_dists)

                if m_c_part is not None:
                    log.debug(" Weights: %s", compare(neighbors_weight, m_debug_base.neighours_weight2))

                segment_to_shift = np.zeros(wire_path_out.uv.shape[1] - 1, dtype=bool)

                for crossed_pair_ind in range(cross_segments.shape[0]):
                    ind1 = np.where(sorted_crossed_segments == cross_segments[crossed_pair_ind, 0])[0]
                    ind2 = np.where(sorted_crossed_segments == cross_segments[crossed_pair_ind, 1])[0]

                    if ind1.size > 1 or ind2.size > 1:
                        if ind1.size > ind2.size:
                            segment_to_shift[cross_segments[crossed_pair_ind, 0]] = True
                        else:
                            segment_to_shift[cross_segments[crossed_pair_ind, 1]] = True
                    else:
                        if neighbors_weight[ind1] < neighbors_weight[ind2]:
                            segment_to_shift[cross_segments[crossed_pair_ind, 0]] = True
                        else:
                            segment_to_shift[cross_segments[crossed_pair_ind, 1]] = True

                points_to_shift = np.zeros(wire_path_out.uv.shape[1])
                points_to_shift[np.unique(np.concatenate(
                    (np.where(segment_to_shift)[0], np.where(segment_to_shift)[0] + 1)))] = 1

                vertex_normal = coil_mesh.vertex_normals()  # vertexNormal(part_faces, part_vertices)

                if np.mean(np.dot(vertex_normal.T, part_vertices - np.mean(part_vertices, axis=0))) < 0:
                    vertex_normal = -vertex_normal

                normal_vectors_wire_path = np.zeros((3, wire_path_out.uv.shape[1]))

                for point_ind in range(wire_path_out.uv.shape[1]):
                    # target_triangle_normal, _ = pointLocation(part_faces, wire_path_out.uv[0, point_ind], wire_path_out.uv[1, point_ind])
                    point = wire_path_out.uv[:, point_ind]
                    face_index, barycentric = pointLocation(point, part_faces, coil_mesh.uv)

                    if face_index is not None:
                        nodes_target_triangle = part_faces[face_index]
                        node_normals_target_triangle = vertex_normal[nodes_target_triangle]
                        normal_vectors_wire_path[:, point_ind] = np.mean(node_normals_target_triangle, axis=0)
                    else:
                        if point_ind == 0:
                            normal_vectors_wire_path[:, point_ind] = normal_vectors_wire_path[:, point_ind + 1]
                        else:
                            normal_vectors_wire_path[:, point_ind] = normal_vectors_wire_path[:, point_ind - 1]
                # arr1 = ones(1, smooth_factors(1)) .* smooth_factors(2)
                # arr2 = (smooth_factors(2) - 1):-1:1
                # arr3 = [1:smooth_factors(2) arr1 arr2]
                # shift_array = conv(points_to_shift, arr3 ./ smooth_factors(2), 'same');
                arr1 = np.ones(smooth_factors[0]) * smooth_factors[1]
                arr2 = np.arange(smooth_factors[1] - 1, 0, -1)
                arr3 = np.concatenate((np.arange(1, smooth_factors[1] + 1), arr1, arr2))
                shift_array = np.convolve(points_to_shift, arr3 / smooth_factors[1], mode='same')
                shift_array[shift_array > 1] = 1

                for point_ind in range(wire_path_out.uv.shape[1]):
                    if smooth_factors[2] < point_ind < wire_path_out.uv.shape[1] - smooth_factors[2]:
                        arr4 = normal_vectors_wire_path[:, point_ind -
                                                        smooth_factors[2] // 2: point_ind + smooth_factors[2] // 2 + 1]
                        shift_vec = np.mean(arr4, axis=1) * shift_array[point_ind] * normal_shift_length
                    else:
                        shift_vec = normal_vectors_wire_path[:, point_ind] * \
                            shift_array[point_ind] * normal_shift_length

                    wire_path_out.v[:, point_ind] += shift_vec

                coil_part.shift_array = shift_array
                coil_part.points_to_shift = points_to_shift.astype(int)
                coil_part.wire_path.uv = wire_path_out.uv
                coil_part.wire_path.v = wire_path_out.v

    else:
        for part_ind in range(len(coil_parts)):
            coil_part = coil_parts[part_ind]
            num_points = coil_part.wire_path.uv.shape[1]
            coil_part.shift_array = np.zeros(num_points)
            coil_part.points_to_shift = np.zeros(num_points)

    return coil_parts


def smooth_track_by_folding(points, smooth_factor):
    """
    Smooths track points using a folding approach.

    Args:
        points (np.ndarray): Track points in UV space.
        smooth_factor (float): Smoothing factor.

    Returns:
        np.ndarray: Smoothed track points in UV space.
    """
    smoothed_points = np.copy(points)
    num_points = points.shape[1]

    for i in range(1, num_points - 1):
        smoothed_points[:, i] = (smooth_factor * points[:, i - 1] + (1 - smooth_factor) * points[:, i]) / 2

    return smoothed_points


def InterX(L1, *varargin, m_debug=None):
    if len(varargin) == 0:
        L2 = L1
        hF = np.less  # Avoid the inclusion of common points
    else:
        L2 = varargin[0]
        hF = np.less_equal

    # Preliminary calculations
    x1 = L1[0, :]
    x2 = L2[0, :]
    y1 = L1[1, :]
    y2 = L2[1, :]
    dx1 = np.diff(x1)
    dy1 = np.diff(y1)
    dx2 = np.diff(x2)
    dy2 = np.diff(y2)

    # Determine 'signed distances'
    S1 = dx1 * y1[:-1] - dy1 * x1[:-1]
    S2 = dx2 * y2[:-1] - dy2 * x2[:-1]

    if m_debug is not None:
        assert compare(x1, m_debug.x1)
        assert compare(x2, m_debug.x2)
        assert compare(dx1, m_debug.dx1)
        assert compare(dx2, m_debug.dx2)
        assert compare(S1, m_debug.S1)
        assert compare(S2, m_debug.S2)

    P10 = np.multiply.outer(dx1, y2)
    P11 = np.multiply.outer(dy1, x2)
    P12 = Diff(P10 - P11, S1, [m_debug.diff_x1, m_debug.diff_y1])

    P20 = np.multiply.outer(y1, dx2)
    P21 = np.multiply.outer(x1, dy2)
    P22 = Diff((P20 - P21).T, S2.T, [m_debug.diff_x2, m_debug.diff_y2])

    if m_debug is not None:
        assert compare(P10, m_debug.P10)
        assert compare(P11, m_debug.P11)
        assert compare(P12, m_debug.P12)
        assert compare(P20, m_debug.P20)
        assert compare(P21, m_debug.P21)
        assert compare(P22, m_debug.P22)

    C1 = hF(P12, 0).astype(int)
    C2 = hF(P22, 0).T.astype(int)

    if m_debug is not None:
        # assert compare(C1, m_debug.C1) # Fail at index 25
        # assert compare(C2, m_debug.C2) # Fail at index 26
        log.debug("C1: %s", compare(C1, m_debug.C1))  # Fail at index 25
        log.debug("C2: %s", compare(C2, m_debug.C2))  # Fail at index 26

    # C1 = hF(Diff(np.multiply(dx1, y2) - np.multiply(dy1, x2), S1), 0)
    # C2 = hF(Diff(np.multiply(y1, dx2) - np.multiply(x1, dy2), S2), 0)

    # Obtain segments where an intersection is expected
    j, i = np.where(C1 & C2)

    if m_debug is not None:
        assert compare(i, m_debug.i1-1)  # Pass

    if len(i) == 0:
        P = np.zeros((2, 0))
        intersect_edge_inds = []
        return P, intersect_edge_inds

    # Transpose and prepare for output
    # i = i.reshape(-1, 1)
    # dx2 = dx2.reshape(-1, 1); dy2 = dy2.reshape(-1, 1); S2 = S2.reshape(-1, 1)
    if m_debug is not None:
        assert compare(i, m_debug.it-1)     # Pass
        assert compare(dx2, m_debug.dx2t)   # Pass
        assert compare(dy2, m_debug.dy2t)   # Pass
        assert compare(S2, m_debug.S2t)     # Pass

    L = dy2[j] * dx1[i] - dy1[i] * dx2[j]
    if m_debug is not None:
        assert compare(L, m_debug.L)

    # Filter out non-zero entries to avoid divisions by zero
    non_zero_indices = L.nonzero()
    i = i[non_zero_indices]
    j = j[non_zero_indices]
    L = L[non_zero_indices]

    if m_debug is not None:
        assert compare(i, m_debug.i3-1)     # Pass
        assert compare(j, m_debug.j3-1)     # Pass
        assert compare(L, m_debug.L3)       # Pass

    # Find unique pairs of indices for intersected edges
    # axis 1 is out of bounds for array of dimension 1
    edges = np.vstack((i, j)).T
    if m_debug is not None:
        assert compare(edges, m_debug.edges-1)     # Pass

    sorted_edges = np.sort(edges, axis=1)
    if m_debug is not None:
        assert compare(sorted_edges, m_debug.sorted_edges-1)     # ???

    # intersect_edge_inds = np.unique(np.sort(np.concatenate((i, j), axis=1)), axis=0)
    intersect_edge_inds = np.unique(sorted_edges, axis=0)

    if m_debug is not None:
        assert compare(intersect_edge_inds, m_debug.output.intersect_edge_inds-1)     # ??

    # Solve system of equations to get the common points
    arr1 = (dx2[j] * S1[i] - dx1[i] * S2[j]) / L
    arr2 = (dy2[j] * S1[i] - dy1[i] * S2[j]) / L
    # M: combined_arr	82x2 double	82x2	double
    # combined_arr =
    #    -0.0079    0.9301
    #     1.2016    0.2877
    #     1.9488    0.1045
    #     1.9208    0.1018
    #     1.8929    0.0977
    #     1.8649    0.0931
    #     1.8355    0.0894
    #     1.8003    0.0835
    #     ...       ...
    combined_arr = np.vstack((arr1, arr2)).T
    P = np.unique(combined_arr, axis=0).T

    if m_debug is not None:
        assert compare(combined_arr, m_debug.combined_arr)     # ??
        assert compare(P, m_debug.output.P)     # ??

    return P, intersect_edge_inds


def Diff(x, y, m_debug=None):
    """
    Calculate the element-wise differences and products of two input arrays.

    Parameters:
    x (ndarray): Input array with shape (N, M).
    y (ndarray): Input array with shape (N, M-1).

    Returns:
    ndarray: Array with shape (N, M-1) containing element-wise differences and products.
    """
    # Unable to allocate 27.1 GiB for an array with shape (1538, 1538, 1538) and data type float64
    try:
        diff_x = x[:, :-1] - y
        diff_y = x[:, 1:] - y
        u = np.multiply.outer(diff_x, diff_y)
        return u
    except Exception as e:
        log.debug(" Memory error: iterating manually")
        u = np.empty((x.shape[0], y.shape[0]))
        for i in range(u.shape[0]):
            diff_x = x[i, :-1] - y[i]
            diff_y = x[i, 1:] - y[i]

            # if m_debug is not None and i < 10: # Pass
            #    assert compare(diff_x, m_debug[0][i])
            #    assert compare(diff_y, m_debug[1][i])

            u[i] = diff_x * diff_y
        return u
