import numpy as np
from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart
from sub_functions.uv_to_xyz import uv_to_xyz

log = logging.getLogger(__name__)

def shift_return_paths(coil_parts: List[CoilPart], input_args):
    """
    Shifts and returns wire paths for coil parts while avoiding wire intersections.

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures containing coil_mesh.
        input_args: Input arguments structure.

    Returns:
        List[CoilPart]: List of CoilPart structures with updated wire paths and shift information.
    """
    
    if not input_args.skip_normal_shift:
        for part_ind in range(len(coil_parts)):
            coil_part = coil_parts[part_ind]
            part_mesh = coil_part.coil_mesh.trimesh_obj
            part_vertices = part_mesh.vertices
            part_faces = part_mesh.faces
            wire_path_out = coil_part.wire_path

            # Delete superimposing points
            points_to_delete = np.concatenate(([True], np.linalg.norm(wire_path_out.v[:, 1:] - wire_path_out.v[:, :-1], axis=0) == 0, [True]))
            wire_path_out.uv[:, points_to_delete] = []
            wire_path_out.v[:, points_to_delete] = []

            if input_args.smooth_flag:
                # Apply smoothing to the wire path
                wire_path_out.uv = smooth_track_by_folding(wire_path_out.uv, input_args.smooth_factor)
                wire_path_out.v, wire_path_out.uv = uv_to_xyz(wire_path_out.uv, part_faces, part_vertices)

            if np.all(wire_path_out.uv[:, -1] == wire_path_out.uv[:, 0]):
                wire_path_out.uv = np.hstack((wire_path_out.uv, wire_path_out.uv[:, :1]))
                wire_path_out.v = np.hstack((wire_path_out.v, wire_path_out.v[:, :1]))

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
                points_to_shift[np.unique(np.concatenate((np.where(segment_to_shift)[0], np.where(segment_to_shift)[0] + 1)))] = 1

                vertex_normal = vertexNormal(part_faces, part_vertices)

                if np.mean(np.dot(vertex_normal, part_vertices - np.mean(part_vertices, axis=0))) < 0:
                    vertex_normal = -vertex_normal

                normal_vectors_wire_path = np.zeros((3, wire_path_out.uv.shape[1]))

                for point_ind in range(wire_path_out.uv.shape[1]):
                    target_triangle_normal, _ = pointLocation(part_faces, wire_path_out.uv[0, point_ind], wire_path_out.uv[1, point_ind])

                    if not np.isnan(target_triangle_normal):
                        nodes_target_triangle = part_faces[target_triangle_normal]
                        node_normals_target_triangle = vertex_normal[nodes_target_triangle]
                        normal_vectors_wire_path[:, point_ind] = np.mean(node_normals_target_triangle, axis=0)
                    else:
                        if point_ind == 0:
                            normal_vectors_wire_path[:, point_ind] = normal_vectors_wire_path[:, point_ind + 1]
                        else:
                            normal_vectors_wire_path[:, point_ind] = normal_vectors_wire_path[:, point_ind - 1]

                shift_array = np.convolve(points_to_shift, np.concatenate(([1], np.ones(input_args.normal_shift_smooth_factors[1]) * input_args.normal_shift_smooth_factors[2], np.arange(input_args.normal_shift_smooth_factors[2], 0, -1) / input_args.normal_shift_smooth_factors[2])) / np.sum(np.concatenate(([1], np.ones(input_args.normal_shift_smooth_factors[1]) * input_args.normal_shift_smooth_factors[2], np.arange(input_args.normal_shift_smooth_factors[2], 0, -1) / input_args.normal_shift_smooth_factors[2]))), mode='same')
                shift_array[shift_array > 1] = 1

                for point_ind in range(wire_path_out.uv.shape[1]):
                    if input_args.normal_shift_smooth_factors[3] < point_ind < wire_path_out.uv.shape[1] - input_args.normal_shift_smooth_factors[3]:
                        shift_vec = np.mean(normal_vectors_wire_path[:, point_ind - input_args.normal_shift_smooth_factors[3] // 2 : point_ind + input_args.normal_shift_smooth_factors[3] // 2 + 1], axis=1) * shift_array[point_ind] * input_args['normal_shift_length']
                    else:
                        shift_vec = normal_vectors_wire_path[:, point_ind] * shift_array[point_ind] * input_args.normal_shift_length

                    wire_path_out.v[:, point_ind] += shift_vec

                coil_part.shift_array = shift_array
                coil_part.points_to_shift = points_to_shift
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

def InterX(L1, *varargin):
    if len(varargin) == 0:
        L2 = L1
        hF = np.less # Avoid the inclusion of common points
    else:
        L2 = varargin[0]
        hF = np.less_equal
    
    # Preliminary calculations
    x1 = L1[0, :]; x2 = L2[0, :]
    y1 = L1[1, :]; y2 = L2[1, :]
    dx1 = np.diff(x1); dy1 = np.diff(y1)
    dx2 = np.diff(x2); dy2 = np.diff(y2)
    
    # Determine 'signed distances'
    S1 = dx1 * y1[:-1] - dy1 * x1[:-1]
    S2 = dx2 * y2[:-1] - dy2 * x2[:-1]
    
    C1 = hF(D(np.multiply.outer(dx1, y2) - np.multiply.outer(dy1, x2), S1), 0)
    C2 = hF(D(np.multiply.outer(y1, dx2) - np.multiply.outer(x1, dy2), S2), 0)
    
    # Obtain segments where an intersection is expected
    i, j = np.where(C1 & C2)
    
    if len(i) == 0:
        P = np.zeros((2, 0))
        intersect_edge_inds = []
        return P, intersect_edge_inds
    
    # Transpose and prepare for output
    i = i.reshape(-1, 1)
    dx2 = dx2.reshape(-1, 1); dy2 = dy2.reshape(-1, 1); S2 = S2.reshape(-1, 1)
    L = dy2[j] * dx1[i] - dy1[i] * dx2[j]
    
    # Filter out non-zero entries to avoid divisions by zero
    non_zero_indices = L.nonzero()
    i = i[non_zero_indices]
    j = j[non_zero_indices]
    L = L[non_zero_indices]
    
    # Find unique pairs of indices for intersected edges
    intersect_edge_inds = np.unique(np.sort(np.concatenate((i, j), axis=1)), axis=0)
    
    # Solve system of equations to get the common points
    P = np.unique(
        np.concatenate((
            (dx2[j] * S1[i] - dx1[i] * S2[j]) / L,
            (dy2[j] * S1[i] - dy1[i] * S2[j]) / L
        ), axis=1),
        axis=0
    )
    
    return P, intersect_edge_inds

def D(x, y):
    return np.multiply.outer(x[:, :-1], y) - np.multiply.outer(y, x[:, 1:])
