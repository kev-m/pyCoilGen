import numpy as np
from scipy.spatial import Delaunay

# Logging
import logging

log = logging.getLogger(__name__)


class triangulation:
    """
    Create an in-memory representation of a 2-D or 3-D triangulation data.

    Args:
        vertices (ndarray): The vertices of the triangulation as a 2-D array.
                          Each row contains the coordinates of a vertex.
        faces (ndarray): The faces of the triangulation as a 2-D array.
                       Each row contains the indices of the vertices forming a face.

    Returns:
        triangulation (ndarray): The triangulation data represented as a 3-D array.
                               Each row contains the coordinates of the three vertices of a triangle.

    Examples:
    >>> vertices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
    >>> triangulation = triangulation(vertices, faces)
    """

    def __init__(self, vertices, faces):
        self._vertices = vertices
        self._faces = faces
        self._tri = None

    def freeBoundary(self):
        if self._tri is None:
            self._tri = Delaunay(self._vertices)
        return self._tri.convex_hull

    def normals(self):
        if self._tri is None:
            self._tri = Delaunay(self._vertices)
        boundary_tri = self._tri
        normals = np.cross(
            boundary_tri.points[boundary_tri.simplices][:, 1] - boundary_tri.points[boundary_tri.simplices][:, 0],
            boundary_tri.points[boundary_tri.simplices][:, 2] - boundary_tri.points[boundary_tri.simplices][:, 0]
        )
        normals /= np.linalg.norm(normals, axis=0, keepdims=True)
        return normals


def freeBoundary(triangulation):
    """
    Find the free boundary facets of the triangles or tetrahedra in a triangulation.

    Args:
        triangulation (ndarray): The triangulation data represented as a 2-D or 3-D array.
                               Each row contains the indices of the vertices forming a triangle or tetrahedron.

    Returns:
        boundaryFacets (ndarray): The free boundary facets of the triangulation.
                               A facet is on the free boundary if it is referenced by only one triangle or tetrahedron.

    Examples:
    >>> triangulation = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    >>> boundaryFacets = freeBoundary(triangulation)
    """

    return triangulation.freeBoundary()


def faceNormal(triangulation):
    """
    Compute the unit normal vectors to all triangles in a 2-D triangulation.

    Args:
        triangulation (ndarray): The triangulation data represented as a 2-D array.
                               Each row contains the indices of the three vertices of a triangle.

    Returns:
        faceNormals (ndarray): A 2-D array where each row contains the unit normal coordinates
                             corresponding to a triangle in the triangulation.

    Note:
        This function supports 2-D triangulations only.

    Examples:
    >>> triangulation = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    >>> faceNormals = faceNormal(triangulation)
    """

    """
    # Compute the centers and face normals of each triangular facet in the boundary triangulation
    normals = np.cross(
        triangulation[:, 1] - triangulation[:, 0],
        triangulation[:, 2] - triangulation[:, 0]
    )
    log.debug("normals: %s, type: %s", normals, normals.dtype)
    #normals /= np.linalg.norm(normals, axis=0, keepdims=True)
    return normals
    """
    return triangulation.normals()


def calculate_face_normals(vertices, faces):
    """
    Calculate face normal vectors given vertices and faces.

    Args:
        vertices (numpy.ndarray): Array of vertex coordinates with shape (N, 3).
        faces (numpy.ndarray): Array of faces defined by vertex indices with shape (M, 3).

    Returns:
        numpy.ndarray: Array of face normal vectors with shape (M, 3).
    """
    # Calculate vectors for each face
    vec1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    vec2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]

    # Compute cross product to get face normals
    normals = np.cross(vec1, vec2)
    log.debug("normals: %s", normals)

    # Normalize face normals
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    return normals
