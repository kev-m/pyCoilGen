import numpy as np

# Logging
import logging

log = logging.getLogger(__name__)


def faceNormal(triangulation):
    """
    Compute the unit normal vectors to all triangles in a 2-D triangulation.

    Parameters:
    - triangulation (ndarray): The triangulation data represented as a 2-D array.
                               Each row contains the indices of the three vertices of a triangle.

    Returns:
    - faceNormals (ndarray): A 2-D array where each row contains the unit normal coordinates
                             corresponding to a triangle in the triangulation.

    Note:
    - This function supports 2-D triangulations only.

    Examples:
    >>> triangulation = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    >>> faceNormals = faceNormal(triangulation)
    """

    log.debug("faceNormal: %s", triangulation.dtype)

    """
    # Calculate the vectors for each triangle edge
    # v1 = triangulation[:, 1, :] - triangulation[:, 0, :]
    # v2 = triangulation[:, 2, :] - triangulation[:, 0, :]
    v1 = triangulation[:, 1] - triangulation[:, 0]
    v2 = triangulation[:, 2] - triangulation[:, 0]

    # Calculate the cross product of the vectors
    crossProduct = np.cross(v1, v2)

    # Calculate the norm of each cross product vector
    norms = np.linalg.norm(crossProduct, axis=0)

    # Calculate the face normals by dividing the cross product vectors by their norms
    faceNormals = crossProduct / norms[:, np.newaxis]

    return faceNormals
    """
    # Compute the centers and face normals of each triangular facet in the boundary triangulation
    normals = np.cross(
        triangulation[:, 1] - triangulation[:, 0],
        triangulation[:, 2] - triangulation[:, 0]
    )
    log.debug("normals: %s, type: %s", normals, normals.dtype)
    #normals /= np.linalg.norm(normals, axis=0, keepdims=True)

    return normals


def triangulation(vertices, faces):
    """
    Create an in-memory representation of a 2-D or 3-D triangulation data.

    Parameters:
    - vertices (ndarray): The vertices of the triangulation as a 2-D array.
                          Each row contains the coordinates of a vertex.
    - faces (ndarray): The faces of the triangulation as a 2-D array.
                       Each row contains the indices of the vertices forming a face.

    Returns:
    - triangulation (ndarray): The triangulation data represented as a 3-D array.
                               Each row contains the coordinates of the three vertices of a triangle.

    Examples:
    >>> vertices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
    >>> triangulation = triangulation(vertices, faces)
    """
    # Create an empty array to store the triangulation data
    triangulation = np.empty((faces.shape[0], 3, vertices.shape[1]), dtype=int)

    # Fill the triangulation array with the coordinates of the triangle vertices
    for i, face in enumerate(faces):
        log.debug("i: %d, face: %s", i, face)
        triangulation[i] = vertices[face]

    return triangulation


def freeBoundary(triangulation):
    """
    Find the free boundary facets of the triangles or tetrahedra in a triangulation.

    Parameters:
    - triangulation (ndarray): The triangulation data represented as a 2-D or 3-D array.
                               Each row contains the indices of the vertices forming a triangle or tetrahedron.

    Returns:
    - boundaryFacets (ndarray): The free boundary facets of the triangulation.
                               A facet is on the free boundary if it is referenced by only one triangle or tetrahedron.

    Examples:
    >>> triangulation = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    >>> boundaryFacets = freeBoundary(triangulation)
    """
    # Convert the triangulation to a set of unique edges
    edges = set()

    for face in triangulation:
        # Generate all possible edges from the face
        face_edges = [(face[i], face[(i + 1) % len(face)])
                      for i in range(len(face))]

        # Add each edge to the set
        edges.update(face_edges)

    # Count the occurrences of each edge
    edge_counts = dict()

    for edge in edges:
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # Find the boundary facets with only one occurrence of each edge
    boundaryFacets = [face for face in triangulation if all(edge_counts[edge] == 1 for edge in [
                                                            (face[i], face[(i + 1) % len(face)]) for i in range(len(face))])]

    return boundaryFacets


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # faceNormal
    triangulations = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    faceNormals = faceNormal(triangulations)
    print("faceNormals", faceNormals)

    # triangulation
    vertices = np.array([[2.5, 8.0],
                         [6.5, 8.0],
                         [2.5, 5.0],
                         [6.5, 5.0],
                         [1.0, 6.5],
                         [8.0, 6.5]])

    faces = np.array([[5, 3, 1],
                      [3, 2, 1],
                      [3, 4, 2],
                      [4, 6, 2]])-1

    triangulations = triangulation(vertices, faces)
    print("triangulation:", triangulations)

    # freeBoundary
    triangulation = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    boundaryFacets = freeBoundary(triangulation)
    print("freeBoundary:", boundaryFacets)
