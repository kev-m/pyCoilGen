"""Independent verification of the magnetic field of a computed coil layout.

The functions here do not use the field calculation of the coil generation pipeline. They evaluate the final wire
path with the closed-form Biot-Savart field of straight wire segments, in double precision, and compare it with the
target field. This answers the question that matters for a user: does the wire that was produced generate the field
that was asked for?

The wire path of a coil part (`coil_part.wire_path.v`, shape (3,n)) is a polyline. It is evaluated for a current of
1 A, summed over all coil parts. The sign of the current in each part was already chosen by the pipeline, by
reversing the direction of the wire path where needed.

Optionally, `biot_savart_magpylib` evaluates the same field with the `magpylib` package, as a cross-check of the
reference implementation. Install it with `pip install pycoilgen[verification]`.
"""

# System
import logging
from typing import Dict, List, Optional

# Third-party
import numpy as np

log = logging.getLogger(__name__)

MU_0 = 4e-7 * np.pi  # Permeability of free space [T m/A]

# Limit for the size of the temporary arrays: points x segments
_MAX_CHUNK_ELEMENTS = 2_000_000


def biot_savart_polyline(wire_points: np.ndarray, field_points: np.ndarray, current: float = 1.0) -> np.ndarray:
    """
    Calculate the magnetic field of a wire polyline, using the closed form field of straight segments.

    For a straight segment from `a` to `b`, with `r1 = p - a` and `r2 = p - b`, the field at `p` is
    `mu0 I / (4 pi) * (r1 x r2) / |r1 x r2|^2 * ((b - a) . (r1/|r1| - r2/|r2|))`.
    The contribution of a segment is zero for points on the line through the segment.

    Args:
        wire_points (np.ndarray): The vertices of the wire, shape (3,n). The wire is not closed automatically.
        field_points (np.ndarray): The points where the field is calculated, shape (3,m).
        current (float): The current in the wire [A].

    Returns:
        np.ndarray: The magnetic field in Tesla, shape (3,m).
    """
    wire_points = np.asarray(wire_points, dtype=np.float64)
    field_points = np.asarray(field_points, dtype=np.float64)
    start = wire_points[:, :-1].T  # (s,3)
    end = wire_points[:, 1:].T
    num_segments = start.shape[0]
    num_points = field_points.shape[1]
    field = np.zeros((num_points, 3))
    if num_segments == 0:
        return field.T

    chunk = max(1, _MAX_CHUNK_ELEMENTS // num_segments)
    direction = end - start
    for first in range(0, num_points, chunk):
        points = field_points[:, first:first + chunk].T  # (c,3)
        r1 = points[:, np.newaxis, :] - start[np.newaxis]  # (c,s,3)
        r2 = points[:, np.newaxis, :] - end[np.newaxis]
        cross = np.cross(r1, r2)
        cross_sq = np.einsum('csk,csk->cs', cross, cross)
        n1 = np.linalg.norm(r1, axis=2)
        n2 = np.linalg.norm(r2, axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_difference = r1 / n1[..., np.newaxis] - r2 / n2[..., np.newaxis]
            factor = np.einsum('sk,csk->cs', direction, unit_difference) / cross_sq
        factor[~np.isfinite(factor)] = 0.0  # Point on the line of the segment, or on its end points
        field[first:first + chunk] = np.einsum('cs,csk->ck', factor, cross)
    return (MU_0 * current / (4.0 * np.pi) * field).T


def biot_savart_magpylib(wire_points: np.ndarray, field_points: np.ndarray, current: float = 1.0) -> np.ndarray:
    """
    Calculate the magnetic field of a wire polyline, using the `magpylib` package.

    This is a cross-check of `biot_savart_polyline`.

    Args:
        wire_points (np.ndarray): The vertices of the wire, shape (3,n).
        field_points (np.ndarray): The points where the field is calculated, shape (3,m).
        current (float): The current in the wire [A].

    Returns:
        np.ndarray: The magnetic field in Tesla, shape (3,m).

    Raises:
        ImportError: If `magpylib` is not installed. Use `pip install pycoilgen[verification]`.
    """
    try:
        import magpylib
    except ImportError as error:
        raise ImportError("magpylib is required for this check: pip install pycoilgen[verification]") from error
    polyline = magpylib.current.Polyline(current=current, vertices=np.asarray(wire_points, dtype=float).T)
    return np.asarray(polyline.getB(np.asarray(field_points, dtype=float).T)).T


def field_of_wire_paths(wire_paths: List[np.ndarray], field_points: np.ndarray, current: float = 1.0,
                        method: str = 'numpy') -> np.ndarray:
    """
    Calculate the total field of several wire paths, each carrying the same current.

    Args:
        wire_paths (list of np.ndarray): The vertices of each wire path, shape (3,n) each.
        field_points (np.ndarray): The points where the field is calculated, shape (3,m).
        current (float): The current in each wire [A].
        method (str): `'numpy'` for the reference implementation, `'magpylib'` for the magpylib cross-check.

    Returns:
        np.ndarray: The magnetic field in Tesla, shape (3,m).

    Raises:
        ValueError: If the method is unknown.
    """
    calculate = {'numpy': biot_savart_polyline, 'magpylib': biot_savart_magpylib}.get(method)
    if calculate is None:
        raise ValueError(f"Unknown method '{method}', use 'numpy' or 'magpylib'.")
    total = np.zeros(np.asarray(field_points).shape)
    for wire_path in wire_paths:
        total += calculate(wire_path, field_points, current)
    return total


def fit_linear_field(field_points: np.ndarray, field_component: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Fit a field component with an affine function of position, `b0 + g . r`, by least squares.

    Args:
        field_points (np.ndarray): The positions, shape (3,m).
        field_component (np.ndarray): The field values at the positions, shape (m,).

    Returns:
        dict: `offset` (float), `gradient` (np.ndarray, shape (3,), per metre) and `residual` (np.ndarray, shape (m,)).
    """
    design = np.hstack([np.ones((field_points.shape[1], 1)), np.asarray(field_points, dtype=float).T])
    solution, *_ = np.linalg.lstsq(design, field_component, rcond=None)
    return {'offset': float(solution[0]), 'gradient': solution[1:], 'residual': field_component - design @ solution}


def verify_field(wire_paths: List[np.ndarray], field_points: np.ndarray, target_field: np.ndarray,
                 design_current: Optional[float] = None, component: int = 2, method: str = 'numpy') -> Dict[str, float]:
    """
    Evaluate how well a wire layout generates a target field, independent of the coil generation pipeline.

    The field of the wire paths is calculated for 1 A, and the current that fits the target best (least squares) is
    determined. All error values are for that best current and are relative to the largest absolute target value
    of the evaluated component, in percent. The gradient is the slope of the fitted field, per Ampere.

    Args:
        wire_paths (list of np.ndarray): The final wire paths of all coil parts, each of shape (3,n).
        field_points (np.ndarray): The target points, shape (3,m).
        target_field (np.ndarray): The target field at the points, shape (3,m), for the current `design_current`.
        design_current (float, optional): The current, in Ampere, for which the target field is given. The best
            current is reported relative to it; if None, the target is taken to be for 1 A.
        component (int): The field component to evaluate, 0, 1 or 2. Default: 2 (the z-component, as in the
            pipeline's field errors).
        method (str): `'numpy'` or `'magpylib'`, see `field_of_wire_paths`.

    Returns:
        dict: The following float values:
            - `best_current`: The current that fits the target best [A].
            - `current_consistency`: `best_current / design_current`; 1.0 if the layout carries the right current.
            - `rms_rel_error`, `max_rel_error`: Error of the field at the best current [%].
            - `rms_rel_error_at_design_current`: RMS error at the design current [%].
            - `efficiency`: The field per Ampere at the target points: the norm of the fitted gradient [T/m/A].
            - `target_gradient`: The norm of the gradient of the target field [T/m], per Ampere of the design current.
            - `linearity_error`: The RMS deviation of the wire field from its linear fit, relative to the largest
              absolute wire field value [%].
            - `field_per_ampere`: The RMS of the field component at the target points [T/A].
    """
    field_points = np.asarray(field_points, dtype=float)
    target = np.asarray(target_field, dtype=float)[component]
    design = 1.0 if design_current is None else float(design_current)

    unit_field = field_of_wire_paths(wire_paths, field_points, 1.0, method)[component]
    denominator = float(np.dot(unit_field, unit_field))
    best_current = float(np.dot(unit_field, target) / denominator) if denominator > 0 else float('nan')
    scale = float(np.max(np.abs(target)))

    error = best_current * unit_field - target
    error_at_design = design * unit_field - target
    wire_fit = fit_linear_field(field_points, unit_field)
    target_fit = fit_linear_field(field_points, target)
    return {
        'best_current': best_current,
        'current_consistency': best_current / design,
        'rms_rel_error': float(np.sqrt(np.mean(error ** 2)) / scale * 100),
        'max_rel_error': float(np.max(np.abs(error)) / scale * 100),
        'rms_rel_error_at_design_current': float(np.sqrt(np.mean(error_at_design ** 2)) / scale * 100),
        'efficiency': float(np.linalg.norm(wire_fit['gradient'])),
        'target_gradient': float(np.linalg.norm(target_fit['gradient'])),
        'linearity_error': float(np.sqrt(np.mean(wire_fit['residual'] ** 2)) / np.max(np.abs(unit_field)) * 100),
        'field_per_ampere': float(np.sqrt(np.mean(unit_field ** 2))),
    }


def verify_solution(solution, component: int = 2, method: str = 'numpy') -> Dict[str, float]:
    """
    Evaluate a computed coil solution with the independent Biot-Savart check.

    The wire paths and the target points are taken from the solution. The pipeline scales the target field to a
    wire current of 1 A per unit of stream function current step (`contour_step`), and stores that in
    `solution.target_field`. The values returned here follow that convention: `best_current` and
    `current_consistency` are relative to the design current (1.0 means the wire path needs exactly the current
    that the stream function was designed for) and `operating_current` is the current in Ampere that the real wire
    needs.

    Args:
        solution (CoilSolution): The solution returned by `pyCoilGen`, with post-processing (wire paths).
        component (int): The field component to evaluate. Default: 2 (z).
        method (str): `'numpy'` or `'magpylib'`.

    Returns:
        dict: The values of `verify_field`, plus `operating_current`: `best_current * contour_step` [A].

    Raises:
        ValueError: If the solution has no wire paths, for example if post-processing was skipped.
    """
    parts = [part for part in solution.coil_parts if part.wire_path is not None]
    if not parts:
        raise ValueError("The solution has no wire paths. Was skip_postprocessing set?")
    result = verify_field([part.wire_path.v for part in parts], solution.target_field.coords,
                          solution.target_field.b, design_current=None, component=component, method=method)
    result['operating_current'] = result['best_current'] * float(solution.coil_parts[0].contour_step)
    return result
