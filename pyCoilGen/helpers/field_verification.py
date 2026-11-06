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


def field_gradient(wire_paths: List[np.ndarray], field_points: np.ndarray, current: float = 1.0, component: int = 2,
                   method: str = 'numpy', step: Optional[float] = None) -> np.ndarray:
    """
    Calculate the spatial gradient of one field component of wire paths, by central differences of the field.

    The field is evaluated with `field_of_wire_paths`, which is independent of the analytic gradient of the pipeline
    (`calculate_gradient`). The default step is a thousandth of the extent of the field points, which is small enough
    for the truncation error to be negligible, and large enough to stay far from rounding errors.

    Args:
        wire_paths (list of np.ndarray): The vertices of each wire path, shape (3,n) each.
        field_points (np.ndarray): The points where the gradient is calculated, shape (3,m).
        current (float): The current in each wire [A].
        component (int): The field component, 0, 1 or 2. Default: 2 (the z-component).
        method (str): `'numpy'` or `'magpylib'`, see `field_of_wire_paths`.
        step (float, optional): The distance of the central differences [m].

    Returns:
        np.ndarray: The gradient of the field component, shape (3,m), in T/m.
    """
    field_points = np.asarray(field_points, dtype=float)
    if step is None:
        step = max(1e-3 * float(np.max(np.ptp(field_points, axis=1))), 1e-9)
    gradient = np.zeros(field_points.shape)
    for axis in range(3):
        offset = np.zeros((3, 1))
        offset[axis] = step
        forward = field_of_wire_paths(wire_paths, field_points + offset, current, method)[component]
        backward = field_of_wire_paths(wire_paths, field_points - offset, current, method)[component]
        gradient[axis] = (forward - backward) / (2.0 * step)
    return gradient


def gradient_deviation(wire_gradient: np.ndarray, target_gradient: np.ndarray) -> Dict[str, float]:
    """
    Quantify how much the gradient of a layout deviates from the desired gradient, over space.

    For imaging, the spatial deviation of the gradient is what distorts the image, not the deviation of the field: a
    uniform offset of the field moves the image, but does not distort it. The layout gradient is scaled with the single
    factor that fits the target gradient best (least squares over all points and all three directions), so that a
    gradient that is a bit too weak, which can be corrected with the current, is not counted. What remains is the
    non-uniformity in strength and direction. The deviation at each point is the norm of the difference between the
    gradient vectors, relative to the mean norm of the target gradient.

    Args:
        wire_gradient (np.ndarray): The gradient of the layout, shape (3,m), for any current.
        target_gradient (np.ndarray): The desired gradient, shape (3,m).

    Returns:
        dict: The following float values:
            - `gradient_scale`: The factor that scales the layout gradient to the target gradient.
            - `gradient_max_dev`, `gradient_rms_dev`: The maximum and RMS deviation over space [%].
            - `gradient_strength`: The mean norm of the scaled layout gradient [T/m]; the target gradient strength
              if the layout is perfect.
    """
    wire_gradient = np.asarray(wire_gradient, dtype=float)
    target_gradient = np.asarray(target_gradient, dtype=float)
    denominator = float(np.sum(wire_gradient * wire_gradient))
    scale = float(np.sum(wire_gradient * target_gradient) / denominator) if denominator > 0 else float('nan')
    nominal = float(np.mean(np.linalg.norm(target_gradient, axis=0)))
    deviation = np.linalg.norm(scale * wire_gradient - target_gradient, axis=0) / nominal * 100
    return {
        'gradient_scale': scale,
        'gradient_max_dev': float(np.max(deviation)),
        'gradient_rms_dev': float(np.sqrt(np.mean(deviation ** 2))),
        'gradient_strength': float(np.mean(np.linalg.norm(scale * wire_gradient, axis=0))),
    }


def verify_field(wire_paths: List[np.ndarray], field_points: np.ndarray, target_field: np.ndarray,
                 design_current: Optional[float] = None, component: int = 2, method: str = 'numpy',
                 target_gradient: Optional[np.ndarray] = None,
                 gradient_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
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
        target_gradient (np.ndarray, optional): The desired gradient of the field component, shape (3,m), for the same
            current as `target_field`. If None, the gradient of the linear fit of the target field is used, which is
            right for a linear target.
        gradient_mask (np.ndarray, optional): Boolean array, shape (m,): the points where the gradient is judged.
            Default: all. Use it to leave out points where the field is suppressed.

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
            - `gradient_max_dev`, `gradient_rms_dev`: The deviation of the gradient over space [%], after scaling
              the gradient to the target, see `gradient_deviation`. Insensitive to a uniform offset of the field.
            - `gradient_strength`: The mean gradient after that scaling [T/m], per Ampere of the design current.
            - `gradient_scale`: The factor of that scaling: the current that fits the target gradient best, like
              `best_current`, but not influenced by the offset of the field.
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

    mask = np.ones(field_points.shape[1], dtype=bool) if gradient_mask is None else np.asarray(gradient_mask, dtype=bool)
    wire_gradient = field_gradient(wire_paths, field_points[:, mask], 1.0, component, method)
    if target_gradient is None:
        reference_gradient = np.tile(target_fit['gradient'][:, np.newaxis], (1, int(mask.sum())))
    else:
        reference_gradient = np.asarray(target_gradient, dtype=float)[:, mask]
    gradient = gradient_deviation(wire_gradient, reference_gradient)
    return {
        **gradient,
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
    contour_step = float(solution.coil_parts[0].contour_step)

    # The desired gradient is known symbolically for the z-component. It is for the same current as the target field
    # before the scaling by the contour step, so scale it like the target field. Points where the field is suppressed
    # have no desired gradient.
    target_gradient, gradient_mask = None, None
    desired_gradient = solution.target_field.target_gradient_dbdxyz
    if component == 2 and desired_gradient is not None:
        target_gradient = np.asarray(desired_gradient, dtype=float) / contour_step
        if solution.is_suppressed_point is not None:
            gradient_mask = ~np.asarray(solution.is_suppressed_point, dtype=bool)

    result = verify_field([part.wire_path.v for part in parts], solution.target_field.coords,
                          solution.target_field.b, design_current=None, component=component, method=method,
                          target_gradient=target_gradient, gradient_mask=gradient_mask)
    result['operating_current'] = result['best_current'] * contour_step
    return result
