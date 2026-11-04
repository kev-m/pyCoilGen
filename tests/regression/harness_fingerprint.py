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
