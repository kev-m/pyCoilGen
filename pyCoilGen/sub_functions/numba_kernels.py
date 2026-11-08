"""
Numba-compiled kernels for the hot paths of the pipeline.

This module imports Numba, so it must only be imported through `accel`, which handles Numba being absent. The kernels
do the same arithmetic as the numpy kernels in `wire_kernels` and `sensitivity_kernels`, but evaluate each (source,
target) pair in registers, in one fused loop that is parallel over the targets (or nodes). The numpy kernels stream
temporary arrays through memory and are limited by memory bandwidth, which these loops are not.

The summation order differs from the numpy kernels, so results agree to rounding error (about 1e-14), not bit for bit.
"""

# Libraries
import numpy as np
import numba as nb

# The Biot-Savart constant, mu_0 / (4 pi)
_C = 1e-7


@nb.njit(parallel=True, cache=True, fastmath=False)
def wire_field(seg_coords, currents, targets):
    """
    The magnetic field of wire segments, for unit current: 1e-7 * sum (I x r) / |r|^3.

    Args:
        seg_coords (ndarray): The centre of each wire segment, (num segments, 3).
        currents (ndarray): The vector of each wire segment, (num segments, 3).
        targets (ndarray): The target points, (num targets, 3).

    Returns:
        ndarray: The field at the target points, (num targets, 3).
    """
    num_segments = seg_coords.shape[0]
    num_targets = targets.shape[0]
    field = np.zeros((num_targets, 3))
    for t in nb.prange(num_targets):
        tx = targets[t, 0]
        ty = targets[t, 1]
        tz = targets[t, 2]
        bx = 0.0
        by = 0.0
        bz = 0.0
        for s in range(num_segments):
            rx = tx - seg_coords[s, 0]
            ry = ty - seg_coords[s, 1]
            rz = tz - seg_coords[s, 2]
            dist2 = rx * rx + ry * ry + rz * rz
            weight = 1.0 / (dist2 * np.sqrt(dist2))
            ix = currents[s, 0]
            iy = currents[s, 1]
            iz = currents[s, 2]
            bx += (iy * rz - iz * ry) * weight
            by += (iz * rx - ix * rz) * weight
            bz += (ix * ry - iy * rx) * weight
        field[t, 0] = bx * _C
        field[t, 1] = by * _C
        field[t, 2] = bz * _C
    return field


@nb.njit(parallel=True, cache=True, fastmath=False)
def wire_gradient(seg_coords, currents, targets):
    """
    The gradient of the magnetic field of wire segments, for unit current.

    Args:
        seg_coords (ndarray): The centre of each wire segment, (num segments, 3).
        currents (ndarray): The vector of each wire segment, (num segments, 3).
        targets (ndarray): The target points, (num targets, 3).

    Returns:
        ndarray: grad[m, k, target] = d B_m / d x_k, (3, 3, num targets).
    """
    num_segments = seg_coords.shape[0]
    num_targets = targets.shape[0]
    grad = np.zeros((3, 3, num_targets))
    for t in nb.prange(num_targets):
        g00 = g01 = g02 = g10 = g11 = g12 = g20 = g21 = g22 = 0.0
        e0 = e1 = e2 = 0.0
        tx = targets[t, 0]
        ty = targets[t, 1]
        tz = targets[t, 2]
        for s in range(num_segments):
            rx = tx - seg_coords[s, 0]
            ry = ty - seg_coords[s, 1]
            rz = tz - seg_coords[s, 2]
            dist2 = rx * rx + ry * ry + rz * rz
            inv_e = 1.0 / (dist2 * np.sqrt(dist2))      # 1 / |r|^3
            theta = -3.0 * _C * inv_e / dist2           # -3 C / |r|^5
            ix = currents[s, 0]
            iy = currents[s, 1]
            iz = currents[s, 2]
            p0 = (iy * rz - iz * ry) * theta            # (I x r), scaled
            p1 = (iz * rx - ix * rz) * theta
            p2 = (ix * ry - iy * rx) * theta
            g00 += p0 * rx
            g01 += p0 * ry
            g02 += p0 * rz
            g10 += p1 * rx
            g11 += p1 * ry
            g12 += p1 * rz
            g20 += p2 * rx
            g21 += p2 * ry
            g22 += p2 * rz
            e0 += ix * inv_e
            e1 += iy * inv_e
            e2 += iz * inv_e
        e0 *= _C
        e1 *= _C
        e2 *= _C
        grad[0, 0, t] = g00
        grad[0, 1, t] = g01 - e2
        grad[0, 2, t] = g02 + e1
        grad[1, 0, t] = g10 + e2
        grad[1, 1, t] = g11
        grad[1, 2, t] = g12 - e0
        grad[2, 0, t] = g20 - e1
        grad[2, 1, t] = g21 + e0
        grad[2, 2, t] = g22
    return grad


@nb.njit(parallel=True, cache=True, fastmath=False)
def node_integral(triangle_points, current, area, node_starts, node_ends, targets, bary, weights, mode):
    """
    Sum the Biot-Savart (mode 0) or the Bz gradient (mode 1) kernel over the Gauss points of the triangles of each node.

    This is the fused equivalent of `sensitivity_kernels.integrate_over_node_triangles` with the kernels of
    `calculate_sensitivity_matrix` and `calculate_gradient_sensitivity_matrix`. The coefficient 1e-7 (and for the
    gradient the plate thickness) is applied by the caller.

    Args:
        triangle_points (ndarray): The triangle corners, (num triangles, 3 corners, 3 xyz), ordered by node.
        current (ndarray): The current density of each triangle, (num triangles, 3).
        area (ndarray): The area of each triangle, (num triangles,).
        node_starts (ndarray): The index of the first triangle of each node, (num nodes,).
        node_ends (ndarray): The index after the last triangle of each node, (num nodes,).
        targets (ndarray): The target points, (num targets, 3).
        bary (ndarray): The barycentric coordinates of the Gauss points, (3, num gauss).
        weights (ndarray): The Gauss weights, (num gauss,).
        mode (int): 0 for the Biot-Savart kernel, 1 for the gradient of Bz.

    Returns:
        ndarray: shape (num nodes, 3, num targets): the sum for each node, component and target point.
    """
    num_nodes = node_starts.shape[0]
    num_targets = targets.shape[0]
    num_gauss = weights.shape[0]
    out = np.zeros((num_nodes, 3, num_targets))
    for n in nb.prange(num_nodes):
        for tri in range(node_starts[n], node_ends[n]):
            ix = current[tri, 0]
            iy = current[tri, 1]
            iz = current[tri, 2]
            for g in range(num_gauss):
                sx = (triangle_points[tri, 0, 0] * bary[0, g] + triangle_points[tri, 1, 0] * bary[1, g]
                      + triangle_points[tri, 2, 0] * bary[2, g])
                sy = (triangle_points[tri, 0, 1] * bary[0, g] + triangle_points[tri, 1, 1] * bary[1, g]
                      + triangle_points[tri, 2, 1] * bary[2, g])
                sz = (triangle_points[tri, 0, 2] * bary[0, g] + triangle_points[tri, 1, 2] * bary[1, g]
                      + triangle_points[tri, 2, 2] * bary[2, g])
                w = weights[g]
                if mode == 0:
                    factor = 2.0 * area[tri] * w
                    for t in range(num_targets):
                        rx = targets[t, 0] - sx
                        ry = targets[t, 1] - sy
                        rz = targets[t, 2] - sz
                        dist2 = rx * rx + ry * ry + rz * rz
                        scale = factor / (dist2 * np.sqrt(dist2))
                        out[n, 0, t] += (iy * rz - iz * ry) * scale
                        out[n, 1, t] += (iz * rx - ix * rz) * scale
                        out[n, 2, t] += (ix * ry - iy * rx) * scale
                else:
                    for t in range(num_targets):
                        rx = targets[t, 0] - sx
                        ry = targets[t, 1] - sy
                        rz = targets[t, 2] - sz
                        dist2 = rx * rx + ry * ry + rz * rz
                        inv_e = 1.0 / (dist2 * np.sqrt(dist2))
                        theta_phi = (ix * ry - iy * rx) * (inv_e / dist2) * (-3.0 * _C)
                        inv_e_c = inv_e * _C
                        out[n, 0, t] += (theta_phi * rx - inv_e_c * iy) * w
                        out[n, 1, t] += (theta_phi * ry + inv_e_c * ix) * w
                        out[n, 2, t] += theta_phi * rz * w
    return out


@nb.njit(parallel=True, cache=True)
def nearest_distances(points_a, points_b):
    """
    For each point of two point sets, the distance to the nearest point of the other set.

    Args:
        points_a (ndarray): The points of the first set, (num a, 3).
        points_b (ndarray): The points of the second set, (num b, 3).

    Returns:
        tuple: The distance of each point of `points_a` to the nearest point of `points_b`, (num a,), and the distance of
            each point of `points_b` to the nearest point of `points_a`, (num b,).
    """
    num_a = points_a.shape[0]
    num_b = points_b.shape[0]
    near_a = np.full(num_a, np.inf)
    near_b = np.full(num_b, np.inf)
    # The nearest squared distance is found first: the square root is monotonic, so the result is the same.
    for a in nb.prange(num_a):
        best = np.inf
        for b in range(num_b):
            dx = points_b[b, 0] - points_a[a, 0]
            dy = points_b[b, 1] - points_a[a, 1]
            dz = points_b[b, 2] - points_a[a, 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best:
                best = d2
        near_a[a] = np.sqrt(best)
    for b in nb.prange(num_b):
        best = np.inf
        for a in range(num_a):
            dx = points_b[b, 0] - points_a[a, 0]
            dy = points_b[b, 1] - points_a[a, 1]
            dz = points_b[b, 2] - points_a[a, 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best:
                best = d2
        near_b[b] = np.sqrt(best)
    return near_a, near_b


@nb.njit(inline='always')
def _crosses(value, strict):
    """The product of the signed distances of the end points of a segment is negative (or zero, if not strict)."""
    if strict:
        return value < 0
    return value <= 0


@nb.njit(parallel=True, cache=True)
def crossing_segment_pairs(x1, y1, x2, y2, dx1, dy1, dx2, dy2, s1, s2, strict):
    """
    The pairs (j, i) of segment j of curve 1 and segment i of curve 2 whose end points lie on opposite sides of the line
    through the other segment (both ways). This is the candidate test of `InterX`, with the same arithmetic.

    Args:
        x1, y1 (ndarray): The points of curve 1, (num points 1,).
        x2, y2 (ndarray): The points of curve 2, (num points 2,).
        dx1, dy1, dx2, dy2 (ndarray): The segment vectors of both curves.
        s1, s2 (ndarray): The signed distances of the segments of both curves.
        strict (bool): True to require a strictly negative product (curve 1 against itself), False to allow zero.

    Returns:
        tuple: j and i, each (num pairs,), ordered by j, then i.
    """
    num_rows = dx1.shape[0]
    num_cols = dx2.shape[0]
    counts = np.zeros(num_rows, dtype=np.int64)
    # Pass 0 counts the pairs of each row; pass 1 stores them at the offset of the row, which keeps the order.
    for j in nb.prange(num_rows):
        count = 0
        side_prev = dx1[j] * y2[0] - dy1[j] * x2[0] - s1[j]
        for i in range(num_cols):
            side_next = dx1[j] * y2[i + 1] - dy1[j] * x2[i + 1] - s1[j]
            if _crosses(side_prev * side_next, strict):
                start = y1[j] * dx2[i] - x1[j] * dy2[i] - s2[i]
                end = y1[j + 1] * dx2[i] - x1[j + 1] * dy2[i] - s2[i]
                if _crosses(start * end, strict):
                    count += 1
            side_prev = side_next
        counts[j] = count
    starts = np.cumsum(counts) - counts
    found_j = np.empty(counts.sum(), dtype=np.int64)
    found_i = np.empty(counts.sum(), dtype=np.int64)
    for j in nb.prange(num_rows):
        count = 0
        side_prev = dx1[j] * y2[0] - dy1[j] * x2[0] - s1[j]
        for i in range(num_cols):
            side_next = dx1[j] * y2[i + 1] - dy1[j] * x2[i + 1] - s1[j]
            if _crosses(side_prev * side_next, strict):
                start = y1[j] * dx2[i] - x1[j] * dy2[i] - s2[i]
                end = y1[j + 1] * dx2[i] - x1[j + 1] * dy2[i] - s2[i]
                if _crosses(start * end, strict):
                    found_j[starts[j] + count] = j
                    found_i[starts[j] + count] = i
                    count += 1
            side_prev = side_next
    return found_j, found_i
