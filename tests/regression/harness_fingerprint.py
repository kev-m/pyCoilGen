"""Fingerprints of coil solutions, and their comparison, for the regression harness.

A fingerprint is a small, JSON-serialisable summary of a `CoilSolution`: counts, lengths, norms and sums of the
important results. It is compact enough to be committed as a golden value, yet sensitive enough to reveal a change in
the mesh, the optimised stream function, the contours, the wire path, the conductor sweep or the field errors.
"""

# System
import math
import platform
from typing import Any, List, Optional

# Third-party
import numpy as np


def summarise_array(array: Optional[np.ndarray]) -> Optional[dict]:
    """
    Summarise an array by its shape, Euclidean norm and sum.

    Args:
        array (np.ndarray or None): The array to summarise.

    Returns:
        dict or None: The summary, or None if there is no array.
    """
    if array is None:
        return None
    array = np.asarray(array, dtype=float)
    return {'shape': list(array.shape), 'norm': float(np.linalg.norm(array)), 'sum': float(np.sum(array))}


def _scalar(value: Any) -> Optional[float]:
    """
    Convert a number, or an array of numbers, to a single float.

    Args:
        value (Any): A scalar, an array, or None.

    Returns:
        float or None: The scalar, the sum of an array, or None.
    """
    if value is None:
        return None
    return float(np.sum(np.asarray(value, dtype=float)))


def fingerprint_part(coil_part) -> dict:
    """
    Create the fingerprint of a single coil part.

    Args:
        coil_part (CoilPart): The coil part of a computed solution.

    Returns:
        dict: The fingerprint.
    """
    result = {
        'mesh_vertices': summarise_array(coil_part.coil_mesh.get_vertices()),
        'mesh_faces': int(len(coil_part.coil_mesh.get_faces())),
        'stream_function': summarise_array(coil_part.stream_function),
        'num_contours': None if coil_part.contour_lines is None else len(coil_part.contour_lines),
        'ohmian_resistance': _scalar(coil_part.ohmian_resistance),
    }

    wire_path = coil_part.wire_path
    if wire_path is None:
        result['wire_path'] = None
    else:
        points = np.asarray(wire_path.v, dtype=float)
        result['wire_path'] = {**summarise_array(points),
                               'length': float(np.sum(np.linalg.norm(np.diff(points, axis=1), axis=0)))}

    sweep = coil_part.layout_surface_mesh
    if sweep is None:
        result['sweep'] = None
    else:
        trimesh_obj = sweep.trimesh_obj
        result['sweep'] = {'vertices': int(len(trimesh_obj.vertices)), 'faces': int(len(trimesh_obj.faces)),
                           'watertight': bool(trimesh_obj.is_watertight),
                           'volume': float(trimesh_obj.volume) if trimesh_obj.is_watertight else None}
    return result


def fingerprint_solution(solution) -> dict:
    """
    Create the fingerprint of a computed solution.

    Args:
        solution (CoilSolution): The solution returned by `pyCoilGen`.

    Returns:
        dict: The fingerprint, with one entry per coil part, the target field and the field errors.
    """
    result = {
        'num_coil_parts': len(solution.coil_parts),
        'coil_parts': [fingerprint_part(part) for part in solution.coil_parts],
        'target_field': summarise_array(solution.target_field.b),
    }

    errors = solution.solution_errors
    if errors is None:
        result['field_errors'] = None
    else:
        values = errors.field_error_vals
        result['field_errors'] = None if values is None else {
            name: _scalar(getattr(values, name)) for name in vars(values)}
        result['opt_current_layout'] = _scalar(errors.opt_current_layout)
        result['combined_field_layout'] = summarise_array(errors.combined_field_layout)
    return result


def environment_info() -> dict:
    """
    Describe the software environment, for recording alongside golden values.

    Returns:
        dict: The versions of Python and the main numerical packages.
    """
    import numpy
    import scipy
    import trimesh
    return {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
            'trimesh': trimesh.__version__}


def compare_fingerprints(expected: Any, actual: Any, rtol: float = 1e-6, atol: float = 1e-9,
                         path: str = '') -> List[str]:
    """
    Compare two fingerprints, and describe every difference.

    Integers, booleans, strings and shapes must match exactly. Floating point numbers must agree within `rtol`
    relative and `atol` absolute tolerance, and NaN only matches NaN.

    Args:
        expected (Any): The golden fingerprint.
        actual (Any): The fingerprint of the current run.
        rtol (float): The relative tolerance for floating point numbers.
        atol (float): The absolute tolerance for floating point numbers.
        path (str): The location within the fingerprint, used in messages.

    Returns:
        list of str: A description of each difference. Empty if the fingerprints match.
    """
    differences = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                differences.append(f"{path}/{key}: missing from the result")
            elif key not in expected:
                differences.append(f"{path}/{key}: not in the golden values")
            else:
                differences += compare_fingerprints(expected[key], actual[key], rtol, atol, f"{path}/{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(f"{path}: length {len(actual)}, expected {len(expected)}")
        else:
            for index, (e, a) in enumerate(zip(expected, actual)):
                differences += compare_fingerprints(e, a, rtol, atol, f"{path}[{index}]")
    elif isinstance(expected, float) or isinstance(actual, float):
        if expected is None or actual is None:
            differences.append(f"{path}: {actual}, expected {expected}")
        elif math.isnan(expected) or math.isnan(actual):
            if not (math.isnan(expected) and math.isnan(actual)):
                differences.append(f"{path}: {actual}, expected {expected}")
        elif not math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol):
            differences.append(f"{path}: {actual!r}, expected {expected!r} "
                               f"(difference {actual - expected:+.3e}, relative {abs(actual - expected) / max(abs(expected), 1e-300):.1e})")
    elif expected != actual:
        differences.append(f"{path}: {actual!r}, expected {expected!r}")
    return differences


# How to judge the physical quality metrics of `pyCoilGen.helpers.field_verification.verify_solution`.
# 'lower' and 'higher' metrics are better when smaller, or larger. 'unity' is better when closer to 1.
# Other metrics (the current, the field per Ampere) are recorded, but not judged.
PHYSICAL_RULES = {
    'gradient_max_dev': 'lower',
    'gradient_rms_dev': 'lower',
    'rms_rel_error': 'lower',
    'max_rel_error': 'lower',
    'rms_rel_error_at_design_current': 'lower',
    'linearity_error': 'lower',
    'current_consistency': 'unity',
    'efficiency': 'higher',
}

# The metrics that decide whether a run is worse. For imaging, the deviation of the gradient over space distorts the
# image, so it is the quality measure; the efficiency is the field per Ampere. The errors of the field itself also
# include a uniform offset of the field, which does not distort an image, so they only give a warning.
PRIMARY_METRICS = ('gradient_max_dev', 'gradient_rms_dev', 'efficiency')

# The absolute part of the tolerance, for metrics in percent, where one percentage point is a meaningful unit.
# Metrics that are not listed use the `atol` argument of `compare_physical`.
ABSOLUTE_TOLERANCE = {
    'gradient_max_dev': 0.5,
    'gradient_rms_dev': 0.5,
}


def compare_physical(golden: dict, actual: dict, rtol: float = 0.05, atol: float = 0.05) -> dict:
    """
    Judge the physical quality metrics of a run against the golden ones: better, equal or worse.

    A metric is 'equal' if it is within `rtol` (relative) plus `atol` (absolute) of the golden value; otherwise it
    is 'better' or 'worse', as defined by `PHYSICAL_RULES`. The absolute tolerance is in the unit of the metric
    (percent for the errors), set per metric for the gradient deviations (`ABSOLUTE_TOLERANCE`, in percentage points),
    0.01 times that for the current consistency, and not used for the efficiency.

    Args:
        golden (dict): The golden metrics.
        actual (dict): The metrics of the current run.
        rtol (float): The relative tolerance.
        atol (float): The absolute tolerance.

    Returns:
        dict: For each judged metric, a tuple of the verdict (`'better'`, `'equal'`, `'worse'`), the golden and the
            actual value.
    """
    verdicts = {}
    for name, rule in PHYSICAL_RULES.items():
        if name not in golden or name not in actual:
            continue
        old, new = golden[name], actual[name]
        if rule == 'unity':
            old, new, rule = abs(old - 1.0), abs(new - 1.0), 'lower'
        slack = rtol * abs(old) + {'current_consistency': 0.01 * atol, 'efficiency': 0.0,
                                   **ABSOLUTE_TOLERANCE}.get(name, atol)
        if abs(new - old) <= slack:
            verdict = 'equal'
        elif (new < old) == (rule == 'lower'):
            verdict = 'better'
        else:
            verdict = 'worse'
        verdicts[name] = (verdict, golden[name], actual[name])
    return verdicts


def worse_metrics(verdicts: dict) -> dict:
    """
    Select the verdicts that make a run worse than the golden values.

    Args:
        verdicts (dict): The result of `compare_physical`.

    Returns:
        dict: The entries of the primary metrics (`PRIMARY_METRICS`) that are 'worse'.
    """
    return {name: values for name, values in verdicts.items() if name in PRIMARY_METRICS and values[0] == 'worse'}


def spec_margin(golden: dict, actual: dict, limit: float, metric: str = 'gradient_max_dev') -> Optional[float]:
    """
    Compute how far a run is within the manufacturing tolerance, and whether it left it.

    A design that was already outside the tolerance in the golden values (for example a coarse test design) has no
    tolerance to keep, so it is only judged relative to the golden values by `compare_physical`.

    Args:
        golden (dict): The golden metrics.
        actual (dict): The metrics of the current run.
        limit (float): The tolerance of the metric, in the unit of the metric (percent).
        metric (str): The metric that the tolerance is about.

    Returns:
        float or None: The margin `limit - actual[metric]`; negative if the run is outside the tolerance. None if the
            golden values were outside the tolerance, or if the metric is missing.
    """
    if metric not in golden or metric not in actual or golden[metric] > limit:
        return None
    return limit - actual[metric]
