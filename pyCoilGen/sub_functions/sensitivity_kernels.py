"""
Vectorised kernels for the sensitivity matrices.

The sensitivity matrices integrate, for every basis function (mesh node) and every target point, the Biot-Savart (or
gradient) kernel over the triangles around the node, with a Gauss-Legendre rule. Doing this node by node, triangle by
triangle and Gauss point by Gauss point in Python is dominated by numpy call overhead on tiny arrays. These helpers do
the same arithmetic on all triangles and Gauss points of a block of nodes at once.
"""

# System
from typing import Callable, List, Tuple

# Logging
import logging

# Libraries
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Local imports
from .data_structures import CoilPart

log = logging.getLogger(__name__)

# The maximum number of (sample point, target point) pairs that are evaluated at once. This is small enough for the
# temporary arrays to stay in the processor cache, which is much faster than streaming them through main memory.
_MAX_PAIRS_PER_BLOCK = 50_000
# The default number of threads. More threads than this do not help: the kernels are limited by memory bandwidth and the
# GIL, not by the arithmetic. Measured on a 16 thread CPU, 2 to 4 threads were fastest and 16 were 20% slower.
_DEFAULT_MAX_THREADS = 4


def resolve_num_threads(requested: int = 0) -> int:
    """
    The number of threads to use for the sensitivity matrices.

    Args:
        requested (int): The `num_threads` input parameter. A positive value is used as given (it may exceed the number
            of CPUs). 0 selects the environment variable PYCOILGEN_THREADS if it is set, otherwise
            min(4, number of CPUs).

    Returns:
        int: The number of threads, at least 1.
    """
    if requested and requested > 0:
        return int(requested)
    env = os.environ.get('PYCOILGEN_THREADS', '')
    if env.strip().isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(_DEFAULT_MAX_THREADS, os.cpu_count() or 1))


def stack_basis_elements(coil_part: CoilPart) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack the per-node triangle data of the basis elements of a coil part into flat arrays.

    The triangles are ordered by node, so the triangles of a node are contiguous.

    Args:
        coil_part (CoilPart): The coil part, with `basis_elements`.

    Returns:
        triangle_points (ndarray): The triangle corners, shape (num triangles, 3 corners, 3 xyz).
        current (ndarray): The current density of each triangle, shape (num triangles, 3).
        area (ndarray): The area of each triangle, shape (num triangles,).
        counts (ndarray): The number of triangles of each node, shape (num nodes,).
    """
    elements = coil_part.basis_elements
    counts = np.array([len(element.area) for element in elements], dtype=int)
    if counts.sum() == 0:
        return np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros(0), counts
    # triangle_points_ABC has the MATLAB layout (triangle, xyz, [node point, point b, point c])
    abc = np.concatenate([np.reshape(element.triangle_points_ABC, (-1, 3, 3)) for element in elements if len(element.area)])
    triangle_points = np.transpose(abc, (0, 2, 1))
    current = np.concatenate([np.reshape(element.current, (-1, 3)) for element in elements if len(element.area)])
    area = np.concatenate([np.reshape(element.area, (-1,)) for element in elements if len(element.area)])
    return triangle_points, current, area, counts


def integrate_over_node_triangles(triangle_points: np.ndarray, current: np.ndarray, area: np.ndarray, counts: np.ndarray,
                                  target_points: np.ndarray, u_coord: np.ndarray, v_coord: np.ndarray,
                                  gauss_weight: np.ndarray,
                                  kernel: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                  num_threads: int = 0) -> np.ndarray:
    """
    Sum a kernel over the Gauss points of all triangles of each node, for all target points.

    Args:
        triangle_points (ndarray): The triangle corners, (num triangles, 3 corners, 3 xyz), ordered by node.
        current (ndarray): The current density of each triangle, (num triangles, 3).
        area (ndarray): The area of each triangle, (num triangles,).
        counts (ndarray): The number of triangles of each node, (num nodes,).
        target_points (ndarray): The target points, (3, num target points).
        u_coord (ndarray): The Gauss point u coordinates, (num gauss,) or (num gauss, 1).
        v_coord (ndarray): The Gauss point v coordinates, same shape as `u_coord`.
        gauss_weight (ndarray): The Gauss point weights, (num gauss,) or (num gauss, 1).
        kernel (callable): kernel(rx, ry, rz, current, area, weight) -> (cx, cy, cz), where `rx`, `ry` and `rz` are the
            components of the vector from the sample point to the target point, each (num samples, num target points),
            and `current`, `area` and `weight` hold the values of the triangle and Gauss point of every sample,
            (num samples, 3), (num samples,) and (num samples,). Each returned component is (num samples,
            num target points). The samples are ordered by triangle, with the Gauss points contiguous:
            sample = triangle * num_gauss + gauss.

        num_threads (int): The number of threads, see `resolve_num_threads`. 0 selects the default.

    Returns:
        ndarray: The sum over each node's samples, shape (3, num target points, num nodes). Nodes without triangles are zero.
    """
    u = np.reshape(u_coord, (-1,))
    v = np.reshape(v_coord, (-1,))
    weights = np.reshape(gauss_weight, (-1,))
    num_gauss = len(u)
    num_nodes = len(counts)
    num_targets = target_points.shape[1]
    result = np.zeros((3, num_targets, num_nodes))
    if len(area) == 0:
        return result

    # Barycentric weights of the Gauss points on (A, B, C)
    bary = np.stack([u, v, 1 - u - v])  # (3, num_gauss)

    node_ends = np.cumsum(counts)
    node_starts = node_ends - counts
    nonempty_nodes = np.nonzero(counts)[0]
    triangles_per_block = max(1, _MAX_PAIRS_PER_BLOCK // max(1, num_gauss * num_targets))

    # Group nodes into blocks, so that each block has about `triangles_per_block` triangles. A node is never split.
    blocks = []
    position = 0
    while position < len(nonempty_nodes):
        last = position
        num_triangles = counts[nonempty_nodes[position]]
        while last + 1 < len(nonempty_nodes) and num_triangles + counts[nonempty_nodes[last + 1]] <= triangles_per_block:
            last += 1
            num_triangles += counts[nonempty_nodes[last]]
        blocks.append(nonempty_nodes[position:last + 1])
        position = last + 1

    def do_block(block_nodes):
        t0 = node_starts[block_nodes[0]]
        t1 = node_ends[block_nodes[-1]]
        points = triangle_points[t0:t1]  # (T, 3 corners, 3 xyz)

        # Sample points: (T, num_gauss, 3)
        samples = np.einsum('tcx,cg->tgx', points, bary)
        samples = samples.reshape(-1, 3)  # (T*G, 3)
        # Components of the vector from each sample to each target: (T*G, P), contiguous
        rx = np.subtract(target_points[0][np.newaxis, :], samples[:, 0:1])
        ry = np.subtract(target_points[1][np.newaxis, :], samples[:, 1:2])
        rz = np.subtract(target_points[2][np.newaxis, :], samples[:, 2:3])

        sample_current = np.repeat(current[t0:t1], num_gauss, axis=0)
        sample_area = np.repeat(area[t0:t1], num_gauss)
        sample_weight = np.tile(weights, t1 - t0)
        components = kernel(rx, ry, rz, sample_current, sample_area, sample_weight)

        # Sum the samples of each node: its triangles are contiguous
        local_starts = (node_starts[block_nodes] - t0) * num_gauss
        for axis, component in enumerate(components):
            result[axis][:, block_nodes] = np.add.reduceat(component, local_starts, axis=0).T  # (P, nodes in block)

    threads = resolve_num_threads(num_threads)
    if threads > 1 and len(blocks) > 1:
        # numpy releases the GIL in the array arithmetic; the blocks write to disjoint columns of `result`
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(do_block, blocks))
    else:
        for block_nodes in blocks:
            do_block(block_nodes)

    return result
