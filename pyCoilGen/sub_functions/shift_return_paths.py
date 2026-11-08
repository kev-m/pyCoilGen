import numpy as np
from typing import List

# Logging
import logging

# Local imports
from .data_structures import CoilPart, Mesh
from .smooth_track_by_folding import smooth_track_by_folding

log = logging.getLogger(__name__)


def shift_return_paths(coil_parts: List[CoilPart], input_args):
    """
    Shifts and returns wire paths for coil parts while avoiding wire intersections.

    Initialises the following properties of a CoilPart:
        - shift_array
        - points_to_shift

    Depends on the following properties of the CoilParts:
        - coil_mesh
        - wire_path

    Depends on the following input_args:
        - skip_normal_shift
        - normal_shift_smooth_factors
        - normal_shift_length
        - smooth_factor

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

            mesh_2d = Mesh(vertices=coil_mesh.uv, faces=part_faces)

            # Delete superimposing points
            diff = np.diff(wire_path_out.v, axis=1)
            points_to_delete = np.concatenate((np.linalg.norm(diff, axis=0), [1]))
            indices_to_delete = np.where(points_to_delete == 0)
            wire_path_out.uv = np.delete(wire_path_out.uv, indices_to_delete, axis=1)
            wire_path_out.v = np.delete(wire_path_out.v, indices_to_delete, axis=1)

            if input_args.smooth_factor > 1:
                # Apply smoothing to the wire path
                wire_path_out.uv = smooth_track_by_folding(wire_path_out.uv, input_args.smooth_factor)
                wire_path_out.v, wire_path_out.uv = coil_mesh.uv_to_xyz(wire_path_out.uv, coil_mesh.uv)

            if np.all(wire_path_out.uv[:, -1] == wire_path_out.uv[:, 0]):
                wire_path_out.add_uv(wire_path_out.uv[:, :1])
                wire_path_out.add_v(wire_path_out.v[:, :1])

            # Detect wire crossings
            cross_points, cross_segments = InterX(wire_path_out.uv)

            if cross_segments.size != 0:
                sorted_crossed_segments = np.sort(np.concatenate((cross_segments[:, 0], cross_segments[:, 1])))
                neighbors_weight = np.zeros(sorted_crossed_segments.shape)
                scale_factor = np.sum(sorted_crossed_segments**4)

                for crossed_seg_ind in range(sorted_crossed_segments.size):
                    check_dists = np.abs(sorted_crossed_segments - sorted_crossed_segments[crossed_seg_ind])
                    check_dists = check_dists[check_dists != 0]
                    neighbors_weight[crossed_seg_ind] = np.min(check_dists)

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

                # Locate all points at once. NOTE: a point that is not on the mesh has face index -1, which selects the
                # last face, as `get_face_index` has always done here.
                face_indices, _ = mesh_2d.get_face_indices(wire_path_out.uv.T)
                node_normals_target_triangles = vertex_normal[part_faces[face_indices]]  # (points, 3 nodes, xyz)
                normal_vectors_wire_path = np.mean(node_normals_target_triangles, axis=1).T

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


# The approximate number of (segment, segment) pairs of InterX that are evaluated at once
_PAIRS_PER_BLOCK = 200_000


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

    # Obtain segments where an intersection is expected: the end points of each segment lie on opposite sides of the
    # line through the other segment (C1: for the segments of L1, C2: for the segments of L2).
    # The (segments x segments) matrices are evaluated in blocks of rows, which gives the same values as the full
    # matrices, but needs much less memory, and the temporary arrays stay in the cache.
    num_rows = dx1.shape[0]
    rows_per_block = max(1, _PAIRS_PER_BLOCK // max(1, x2.shape[0]))
    found_j, found_i = [], []
    for first in range(0, num_rows, rows_per_block):
        rows = slice(first, min(first + rows_per_block, num_rows))
        # C1: P12[a, b] = (X[a, b] - S1[a]) * (X[a, b + 1] - S1[a]), X = dx1 y2 - dy1 x2
        side = np.multiply.outer(dx1[rows], y2) - np.multiply.outer(dy1[rows], x2)
        side -= S1[rows, np.newaxis]
        c1 = hF(side[:, :-1] * side[:, 1:], 0)
        # C2: P22[b, a] = (Y[b, a] - S2[b]) * (Y[b, a + 1] - S2[b]), Y = y1 dx2 - x1 dy2. Only the few pairs where C1 holds
        # need to be tested, with the same arithmetic as for the full matrix. They stay in the order of `np.where`.
        block_j, block_i = np.nonzero(c1)
        row = block_j + first
        side_start = y1[row] * dx2[block_i] - x1[row] * dy2[block_i] - S2[block_i]
        side_end = y1[row + 1] * dx2[block_i] - x1[row + 1] * dy2[block_i] - S2[block_i]
        keep = hF(side_start * side_end, 0)
        block_j, block_i = block_j[keep], block_i[keep]
        found_j.append(block_j + first)
        found_i.append(block_i)
    j = np.concatenate(found_j) if found_j else np.zeros(0, dtype=int)
    i = np.concatenate(found_i) if found_i else np.zeros(0, dtype=int)

    if len(i) == 0:
        P = np.zeros((2, 0))
        intersect_edge_inds = []
        return P, intersect_edge_inds

    # Transpose and prepare for output
    # i = i.reshape(-1, 1)
    # dx2 = dx2.reshape(-1, 1); dy2 = dy2.reshape(-1, 1); S2 = S2.reshape(-1, 1)
    L = dy2[j] * dx1[i] - dy1[i] * dx2[j]

    # Filter out non-zero entries to avoid divisions by zero
    non_zero_indices = L.nonzero()
    i = i[non_zero_indices]
    j = j[non_zero_indices]
    L = L[non_zero_indices]

    # Find unique pairs of indices for intersected edges
    # axis 1 is out of bounds for array of dimension 1
    edges = np.vstack((i, j)).T
    sorted_edges = np.sort(edges, axis=1)

    # intersect_edge_inds = np.unique(np.sort(np.concatenate((i, j), axis=1)), axis=0)
    intersect_edge_inds = np.unique(sorted_edges, axis=0)

    # Solve system of equations to get the common points
    arr1 = (dx2[j] * S1[i] - dx1[i] * S2[j]) / L
    arr2 = (dy2[j] * S1[i] - dy1[i] * S2[j]) / L
    combined_arr = np.vstack((arr1, arr2)).T
    P = np.unique(combined_arr, axis=0).T

    return P, intersect_edge_inds


def Diff(x, y):
    """
    Calculate the element-wise differences and products of two input arrays.

    Parameters:
    x (ndarray): Input array with shape (N, M).
    y (ndarray): Input array with shape (N, M-1).

    Returns:
    ndarray: Array with shape (N, M-1) containing element-wise differences and products.
    """
    return (x[:, :-1] - y[:, np.newaxis]) * (x[:, 1:] - y[:, np.newaxis])
