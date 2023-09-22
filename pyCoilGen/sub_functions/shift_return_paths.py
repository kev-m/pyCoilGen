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

                for point_ind in range(wire_path_out.uv.shape[1]):
                    point = wire_path_out.uv[:, point_ind]
                    face_index, barycentric = mesh_2d.get_face_index(point)

                    if face_index is not None:
                        nodes_target_triangle = part_faces[face_index]
                        node_normals_target_triangle = vertex_normal[nodes_target_triangle]
                        normal_vectors_wire_path[:, point_ind] = np.mean(node_normals_target_triangle, axis=0)
                    else:
                        if point_ind == 0:
                            normal_vectors_wire_path[:, point_ind] = normal_vectors_wire_path[:, point_ind + 1]
                        else:
                            normal_vectors_wire_path[:, point_ind] = normal_vectors_wire_path[:, point_ind - 1]

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

    P10 = np.multiply.outer(dx1, y2)
    P11 = np.multiply.outer(dy1, x2)
    if m_debug is not None:
        P12 = Diff(P10 - P11, S1, [m_debug.diff_x1, m_debug.diff_y1])
    else:
        P12 = Diff(P10 - P11, S1)

    P20 = np.multiply.outer(y1, dx2)
    P21 = np.multiply.outer(x1, dy2)
    if m_debug is not None:
        P22 = Diff((P20 - P21).T, S2.T, [m_debug.diff_x2, m_debug.diff_y2])
    else:
        P22 = Diff((P20 - P21).T, S2.T)

    C1 = hF(P12, 0).astype(int)
    C2 = hF(P22, 0).T.astype(int)

    # Obtain segments where an intersection is expected
    j, i = np.where(C1 & C2)

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
    u = np.empty((x.shape[0], y.shape[0]))
    for i in range(u.shape[0]):
        diff_x = x[i, :-1] - y[i]
        diff_y = x[i, 1:] - y[i]

        u[i] = diff_x * diff_y
    return u
