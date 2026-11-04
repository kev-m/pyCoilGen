"""Helpers for treating a planar coil part as a flat layer, e.g. for fabrication exports."""

# System
import numpy as np
from typing import Tuple

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilPart

# The maximum allowed out-of-plane spread of a coil part, relative to its in-plane extent.
PLANARITY_TOLERANCE = 0.1


def get_viewing_basis(coil_part: CoilPart) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the orthonormal basis used to project a planar coil part onto a flat layer.

    The coil part is viewed from the side that its coil mesh face normals point towards. For a bi-planar coil, whose
    plates have outward pointing normals, each plate is therefore seen from the outside, as it would be viewed when
    laid face-up on a bench. Looking along the negative normal, `u` points to the right and `w` points up, so the
    projection is never a mirror image of the view.

    The first basis vector, `u`, is the projection of the global x axis onto the plane (or the global y axis, if the
    plane is perpendicular to x).

    Args:
        coil_part (CoilPart): The coil part. Its `coil_mesh` must have consistently oriented faces.

    Returns:
        tuple of np.ndarray: The unit vectors (u, w, n), forming a right-handed system, where n is the viewing normal.

    Raises:
        ValueError: If the coil mesh face normals cancel out, so that no viewing direction can be determined.
    """
    trimesh_obj = coil_part.coil_mesh.trimesh_obj
    normal = np.sum(trimesh_obj.face_normals * trimesh_obj.area_faces[:, np.newaxis], axis=0)
    length = np.linalg.norm(normal)
    if length < 1e-12 * np.sum(trimesh_obj.area_faces):
        raise ValueError("The coil mesh face normals cancel out; unable to determine the viewing direction.")
    normal = normal / length

    reference = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = reference - np.dot(reference, normal) * normal
    u = u / np.linalg.norm(u)
    w = np.cross(normal, u)
    return u, w, normal


def project_to_plane(points: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Project 3D points onto a plane, converting from metres to millimetres.

    Args:
        points (np.ndarray): A (3, n) or (n, 3) array of points, in metres.
        u (np.ndarray): The unit vector along the plane x axis.
        w (np.ndarray): The unit vector along the plane y axis.

    Returns:
        np.ndarray: An (n, 2) array of in-plane co-ordinates, in millimetres.
    """
    points = np.asarray(points)
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T
    return 1000.0 * np.column_stack((points @ u, points @ w))


def check_planar(coil_part: CoilPart, normal: np.ndarray):
    """
    Check that a coil part is (nearly) planar, as required for a flat layer.

    Shifts of the wire path along the surface normal, e.g. from `normal_shift_length`, are permitted: they are small
    compared with the extent of the coil part.

    Args:
        coil_part (CoilPart): The coil part.
        normal (np.ndarray): The unit viewing normal.

    Raises:
        ValueError: If the spread of the coil mesh along the normal is large, compared to its in-plane extent.
    """
    vertices = coil_part.coil_mesh.get_vertices()
    out_of_plane = np.ptp(vertices @ normal)
    in_plane = np.max(np.ptp(vertices, axis=0))
    if out_of_plane > PLANARITY_TOLERANCE * in_plane:
        raise ValueError("Only planar coil parts are supported. "
                         f"The coil surface deviates {out_of_plane * 1000:.1f} mm from a plane, over an extent of "
                         f"{in_plane * 1000:.1f} mm.")
