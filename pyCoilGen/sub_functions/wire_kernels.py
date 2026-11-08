"""
Vectorised Biot-Savart kernels for a wire that is given as a sequence of straight segments.

For every target point, the field (or its gradient) is the sum of the contributions of all wire segments. This is done
in blocks of (target points x segments) that are small enough for the temporary arrays to stay in the processor
cache, with contiguous per-component arrays and in-place operations. That is several times faster than building the
full (segments x targets x 3) arrays, as the memory traffic of those dominates the arithmetic.

For large problems, the compiled kernels of `numba_kernels` are used when Numba is installed (see `accel`). The numpy
kernels here (`biot_savart_field_numpy` and `biot_savart_gradient_numpy`) are the reference that they are tested against.
"""

# Libraries
import numpy as np

# Local imports
from . import accel

# The approximate number of (segment, target) pairs that are evaluated at once.
_PAIRS_PER_BLOCK = 32_768
# The maximum number of target points of a block.
_MAX_TARGETS_PER_BLOCK = 512


def _blocks(num_segments: int, num_targets: int):
    """Yield (target slice, segment slice) pairs that cover all (target, segment) combinations."""
    targets_per_block = max(1, min(num_targets, _MAX_TARGETS_PER_BLOCK))
    segments_per_block = max(1, _PAIRS_PER_BLOCK // targets_per_block)
    for t0 in range(0, num_targets, targets_per_block):
        for s0 in range(0, num_segments, segments_per_block):
            yield slice(t0, t0 + targets_per_block), slice(s0, s0 + segments_per_block)


def _relative_vectors(seg_coords: np.ndarray, targets: np.ndarray):
    """The components of the vectors from the segments to the targets, each (segments, targets) and contiguous."""
    rx = np.subtract(targets[np.newaxis, :, 0], seg_coords[:, 0:1])
    ry = np.subtract(targets[np.newaxis, :, 1], seg_coords[:, 1:2])
    rz = np.subtract(targets[np.newaxis, :, 2], seg_coords[:, 2:3])
    return rx, ry, rz


def biot_savart_field_numpy(seg_coords: np.ndarray, currents: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    The magnetic field of wire segments, by the Biot-Savart law, for unit current: 1e-7 * sum (I x r) / |r|^3.

    Args:
        seg_coords (ndarray): The centre of each wire segment, (num segments, 3).
        currents (ndarray): The vector of each wire segment (the current direction times its length), (num segments, 3).
        targets (ndarray): The target points, (num targets, 3).

    Returns:
        ndarray: The field at the target points, (num targets, 3).
    """
    seg_coords = np.asarray(seg_coords, dtype=float)
    currents = np.asarray(currents, dtype=float)
    targets = np.asarray(targets, dtype=float)
    field = np.zeros((targets.shape[0], 3))
    for target_slice, segment_slice in _blocks(seg_coords.shape[0], targets.shape[0]):
        rx, ry, rz = _relative_vectors(seg_coords[segment_slice], targets[target_slice])
        dist2 = rx * rx
        dist2 += ry * ry
        dist2 += rz * rz
        inv = np.sqrt(dist2)
        inv *= dist2
        np.reciprocal(inv, out=inv)  # 1 / |r|^3
        rx *= inv
        ry *= inv
        rz *= inv
        # Sum over the segments of I_a * (r_b / |r|^3), with BLAS: moment[a, target] for each component b
        current_t = currents[segment_slice].T
        mx, my, mz = current_t @ rx, current_t @ ry, current_t @ rz  # (3, targets)
        # B = sum I x r / |r|^3
        field[target_slice, 0] += mz[1] - my[2]
        field[target_slice, 1] += mx[2] - mz[0]
        field[target_slice, 2] += my[0] - mx[1]
    return 1e-7 * field


def biot_savart_gradient_numpy(seg_coords: np.ndarray, currents: np.ndarray, targets: np.ndarray):
    """
    The gradient of the magnetic field of wire segments, by the Biot-Savart law, for unit current.

    Args:
        seg_coords (ndarray): The centre of each wire segment, (num segments, 3).
        currents (ndarray): The vector of each wire segment (the current direction times its length), (num segments, 3).
        targets (ndarray): The target points, (num targets, 3).

    Returns:
        tuple: dBx/dxyz, dBy/dxyz and dBz/dxyz, each (num targets, 3): the derivatives of the field component with respect to x, y and z.
    """
    seg_coords = np.asarray(seg_coords, dtype=float)
    currents = np.asarray(currents, dtype=float)
    targets = np.asarray(targets, dtype=float)
    num_targets = targets.shape[0]
    # grad[m, k, target] = d B_m / d x_k
    grad = np.zeros((3, 3, num_targets))
    C = 1e-7
    for target_slice, segment_slice in _blocks(seg_coords.shape[0], num_targets):
        r = _relative_vectors(seg_coords[segment_slice], targets[target_slice])
        rx, ry, rz = r
        i = currents[segment_slice]
        ix, iy, iz = i[:, 0:1], i[:, 1:2], i[:, 2:3]

        inv = rx * rx
        inv += ry * ry
        inv += rz * rz
        np.sqrt(inv, out=inv)
        np.reciprocal(inv, out=inv)  # 1 / |r|
        inv2 = inv * inv
        inv_e = inv2 * inv  # 1 / |r|^3
        inv2 *= inv2
        inv2 *= inv  # 1 / |r|^5
        inv2 *= -3 * C  # theta' = -3 C / |r|^5, so that dB_m/dx_k = theta' r_k phi_m + C / |r|^3 * (terms of I)

        phi = (iy * rz - iz * ry, iz * rx - ix * rz, ix * ry - iy * rx)  # I x r
        for k in range(3):
            theta_r = inv2 * r[k]
            for m in range(3):
                grad[m, k, target_slice] += np.einsum('sp,sp->p', theta_r, phi[m])

        # The terms of I: sum of C / |r|^3 * I_a over the segments
        sum_ie = (i.T @ inv_e) * C  # (3, targets): sum_s I_a / |r|^3
        grad[0, 1, target_slice] -= sum_ie[2]
        grad[0, 2, target_slice] += sum_ie[1]
        grad[1, 0, target_slice] += sum_ie[2]
        grad[1, 2, target_slice] -= sum_ie[0]
        grad[2, 0, target_slice] -= sum_ie[1]
        grad[2, 1, target_slice] += sum_ie[0]
    return grad[0].T.copy(), grad[1].T.copy(), grad[2].T.copy()


def _as_contiguous(*arrays):
    """The arrays as contiguous float64, so that the compiled kernels are compiled once, for one layout."""
    return tuple(np.ascontiguousarray(a, dtype=np.float64) for a in arrays)


def biot_savart_field(seg_coords: np.ndarray, currents: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    The magnetic field of wire segments, by the Biot-Savart law, for unit current: 1e-7 * sum (I x r) / |r|^3.

    Uses the compiled kernel if Numba is installed and the problem is large, otherwise the numpy kernel.

    Args:
        seg_coords (ndarray): The centre of each wire segment, (num segments, 3).
        currents (ndarray): The vector of each wire segment (the current direction times its length), (num segments, 3).
        targets (ndarray): The target points, (num targets, 3).

    Returns:
        ndarray: The field at the target points, (num targets, 3).
    """
    kernels = accel.kernels_for(len(seg_coords) * len(targets))
    if kernels is None:
        return biot_savart_field_numpy(seg_coords, currents, targets)
    accel.set_threads()
    return kernels.wire_field(*_as_contiguous(seg_coords, currents, targets))


def biot_savart_gradient(seg_coords: np.ndarray, currents: np.ndarray, targets: np.ndarray):
    """
    The gradient of the magnetic field of wire segments, by the Biot-Savart law, for unit current.

    Uses the compiled kernel if Numba is installed and the problem is large, otherwise the numpy kernel.

    Args:
        seg_coords (ndarray): The centre of each wire segment, (num segments, 3).
        currents (ndarray): The vector of each wire segment (the current direction times its length), (num segments, 3).
        targets (ndarray): The target points, (num targets, 3).

    Returns:
        tuple: dBx/dxyz, dBy/dxyz and dBz/dxyz, each (num targets, 3): the derivatives of the field component with respect to x, y and z.
    """
    kernels = accel.kernels_for(len(seg_coords) * len(targets))
    if kernels is None:
        return biot_savart_gradient_numpy(seg_coords, currents, targets)
    accel.set_threads()
    grad = kernels.wire_gradient(*_as_contiguous(seg_coords, currents, targets))
    return grad[0].T.copy(), grad[1].T.copy(), grad[2].T.copy()
