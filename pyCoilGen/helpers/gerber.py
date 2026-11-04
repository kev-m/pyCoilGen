"""Helpers for creating Gerber (RS-274X) files from planar polylines.

All co-ordinates passed to these functions are in millimetres. The generated files use millimetre units and the
`4.6` co-ordinate format (4 integer digits, 6 decimal digits), with leading zeros omitted.
"""
# System
from typing import List, Optional, Sequence

# Third-party
import numpy as np

# Number of decimal digits in the Gerber co-ordinate format (`%FSLAX46Y46*%`).
_DECIMALS = 6
_SCALE = 10 ** _DECIMALS

# The aperture used for all tracks.
_TRACK_APERTURE = 10


def gerber_coordinate(x: float, y: float, operation: str) -> str:
    """
    Format a single Gerber co-ordinate command.

    Args:
        x (float): X co-ordinate, in millimetres.
        y (float): Y co-ordinate, in millimetres.
        operation (str): The Gerber operation code, one of 'D01' (draw), 'D02' (move) or 'D03' (flash).

    Returns:
        str: The formatted command, e.g. 'X1500000Y-2250000D01*'.

    Raises:
        ValueError: If the operation is not supported, or the co-ordinates do not fit the `4.6` format.
    """
    if operation not in ('D01', 'D02', 'D03'):
        raise ValueError(f"Unsupported Gerber operation '{operation}'.")
    if abs(x) >= 10000 or abs(y) >= 10000:
        raise ValueError(f"Co-ordinate ({x}, {y}) mm does not fit the Gerber 4.6 format (limit 9999 mm).")
    return f"X{int(round(x * _SCALE))}Y{int(round(y * _SCALE))}{operation}*"


def _remove_repeated_points(points: np.ndarray) -> np.ndarray:
    """
    Remove consecutive points that map to the same Gerber co-ordinate.

    Args:
        points (np.ndarray): An (n, 2) array of points, in millimetres.

    Returns:
        np.ndarray: The points, without consecutive duplicates.
    """
    scaled = np.round(points * _SCALE)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(np.diff(scaled, axis=0) != 0, axis=1)
    return points[keep]


def _stroke_lines(path: np.ndarray) -> List[str]:
    """
    Create the commands to stroke a polyline: move to the first point, then draw to each subsequent point.

    Args:
        path (np.ndarray): An (n, 2) array of points, in millimetres.

    Returns:
        list of str: The Gerber commands. Empty if the path has fewer than two distinct points.
    """
    path = _remove_repeated_points(np.asarray(path, dtype=float))
    if len(path) < 2:
        return []
    lines = [gerber_coordinate(path[0, 0], path[0, 1], 'D02')]
    lines += [gerber_coordinate(x, y, 'D01') for x, y in path[1:]]
    return lines


def _region_lines(contour: np.ndarray) -> List[str]:
    """
    Create the commands for a filled region bounded by a closed contour.

    Args:
        contour (np.ndarray): An (n, 2) array of points, in millimetres. The contour is closed automatically.

    Returns:
        list of str: The Gerber commands, from 'G36*' to 'G37*'.

    Raises:
        ValueError: If the contour has fewer than three distinct points.
    """
    contour = _remove_repeated_points(np.asarray(contour, dtype=float))
    if len(contour) >= 2 and np.all(np.round(contour[0] * _SCALE) == np.round(contour[-1] * _SCALE)):
        contour = contour[:-1]
    if len(contour) < 3:
        raise ValueError("A region needs at least three distinct points.")
    lines = ['G36*', gerber_coordinate(contour[0, 0], contour[0, 1], 'D02')]
    lines += [gerber_coordinate(x, y, 'D01') for x, y in contour[1:]]
    lines += [gerber_coordinate(contour[0, 0], contour[0, 1], 'D01'), 'G37*']
    return lines


def _header(comment: str, track_width: float) -> List[str]:
    """
    Create the Gerber file header.

    Args:
        comment (str): A comment describing the file. Any '*' characters are removed.
        track_width (float): The track (aperture) width, in millimetres.

    Returns:
        list of str: The header commands.

    Raises:
        ValueError: If the track width is not positive.
    """
    if track_width <= 0:
        raise ValueError("The track width must be positive.")
    comment = comment.replace('*', ' ')
    return [
        f"G04 {comment}*",
        "%MOMM*%",
        f"%FSLAX{4}{_DECIMALS}Y{4}{_DECIMALS}*%",
        f"%ADD{_TRACK_APERTURE}C,{track_width:.6f}*%",
        "G01*",
    ]


def create_gerber_tracks(paths: Sequence[np.ndarray], track_width: float,
                         comment: str = "pyCoilGen coil tracks") -> str:
    """
    Create the contents of a Gerber file that draws polylines as tracks of a constant width.

    Args:
        paths (sequence of np.ndarray): Each path is an (n, 2) array of points, in millimetres.
        track_width (float): The track width, in millimetres.
        comment (str, optional): A comment to embed in the file.

    Returns:
        str: The Gerber file contents.

    Raises:
        ValueError: If the track width is not positive, or a co-ordinate does not fit the Gerber format.

    Example:
        >>> text = create_gerber_tracks([np.array([[0.0, 0.0], [10.0, 0.0]])], 2.0)
        >>> 'X10000000Y0D01*' in text
        True
    """
    lines = _header(comment, track_width)
    lines += ["%LPD*%", f"D{_TRACK_APERTURE}*"]
    for path in paths:
        lines += _stroke_lines(path)
    lines.append("M02*")
    return "\n".join(lines) + "\n"


def create_gerber_complement(outline: np.ndarray, paths: Sequence[np.ndarray], track_width: float,
                             holes: Optional[Sequence[np.ndarray]] = None,
                             comment: str = "pyCoilGen coil complement") -> str:
    """
    Create the contents of a Gerber file for the complement of the tracks: a solid region bounded by the outline,
    with the tracks (and any holes) cleared.

    This is the copper that remains when the tracks are cut out of a solid plate.

    Args:
        outline (np.ndarray): An (n, 2) array of points, in millimetres, of the outer boundary. Closed automatically.
        paths (sequence of np.ndarray): The tracks to clear. Each path is an (n, 2) array of points, in millimetres.
        track_width (float): The width of the cleared tracks, in millimetres.
        holes (sequence of np.ndarray, optional): Inner boundaries to clear, each an (n, 2) array of points.
        comment (str, optional): A comment to embed in the file.

    Returns:
        str: The Gerber file contents.

    Raises:
        ValueError: If the track width is not positive, a boundary is degenerate, or a co-ordinate does not fit the
            Gerber format.
    """
    lines = _header(comment, track_width)
    lines += ["%LPD*%"] + _region_lines(outline)
    lines += ["%LPC*%"]
    for hole in (holes or []):
        lines += _region_lines(hole)
    lines += [f"D{_TRACK_APERTURE}*"]
    for path in paths:
        lines += _stroke_lines(path)
    lines.append("M02*")
    return "\n".join(lines) + "\n"
