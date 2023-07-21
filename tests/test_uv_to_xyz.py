# Hack code
from trimesh import Trimesh
from trimesh.proximity import ProximityQuery
from trimesh.points import point_plane_distance
import numpy as np


# Set up paths: Add the project root directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Test support
from sub_functions.build_planar_mesh import build_planar_mesh
from sub_functions.data_structures import Mesh
# Code under test
from sub_functions.uv_to_xyz import uv_to_xyz


####### Possible alternate implementation ######
def point_inside_triangleX(point, triangle_vertices):
    """
    Check if a 2D point is contained on or in a triangle.

    Args:
        point (ndarray): 2D point as a 2-element array [x, y].
        triangle_vertices (ndarray): Triangle vertices as a 3x2 array [[x1, y1], [x2, y2], [x3, y3]].

    Returns:
        bool: True if the point is inside or on the triangle, False otherwise.
    """
    x, y = point
    x1, y1 = triangle_vertices[0]
    x2, y2 = triangle_vertices[1]
    x3, y3 = triangle_vertices[2]

    # Calculate the barycentric coordinates
    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    gamma = 1 - alpha - beta

    # Check if the point is inside the triangle
    return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1


def which_faceX(point, face_indices, face_vertices):
    """
    Determine which face contains the point.

    Args:
        point (xyz): The input 2D points with shape (2,n).
        face_indices (ndarray): The indices of the possible faces.
        face_vertices (Trimesh): The vertices of the possible faces.

    Returns:
        index (int): The index of the possible face or None if the point intersects multiple faces.
    """
    results = [point_inside_triangle(point[:2], face_vertex[:,:2]) for face_vertex in face_vertices]
    if np.sum(results) != 1:
        log.debug(" Unable to match point %s to face: %d matches <- %s", point, np.sum(results), face_indices)
        return None
    return face_indices[np.where(results)][0]


def get_face(point, planary_mesh: Trimesh):
    """
    Get the face that contains the given point.

    Args:
        point (ndarray): The input 2D point (x,y,0)
        planary_mesh (Trimesh): The 3D Trimesh representation of the 2D mesh (x,y,0)

    Returns:
        face (int): The index of the triangle that contains the point else None

    """
    proximity = ProximityQuery(planary_mesh)
    distance, vertex_id = proximity.vertex(point)
    possible_triangles = planary_mesh.vertex_faces[vertex_id]
    refined_triangles = possible_triangles[np.where(possible_triangles >= 0)]
    if len(refined_triangles) > 0:
        planar_vertices = planary_mesh.vertices.view(np.ndarray)
        target_triangle = which_face(point, refined_triangles, planar_vertices[planary_mesh.faces[refined_triangles]])
        return target_triangle
    log.debug("Unable to find any face for point %s", point)
    return None


################################################
################################################
def barycentric_coordinates(point, triangle_vertices):
    """
    Calculate the barycentric coordinates of a 2D point inside a triangle.

    Args:
        point (ndarray): The 2D point as a 1x2 array [x, y].
        triangle_vertices (ndarray): The vertices of the triangle as a 3x2 array.

    Returns:
        ndarray: The barycentric coordinates of the point as a 1x3 array [alpha, beta, gamma].
    """
    # Extract the coordinates of the point and triangle vertices
    x, y = point
    x1, y1 = triangle_vertices[0]
    x2, y2 = triangle_vertices[1]
    x3, y3 = triangle_vertices[2]

    # Calculate the areas of sub-triangles
    area_triangle = 0.5 * (-y2 * x3 + y1 * (-x2 + x3) + x1 * (y2 - y3) + x2 * y3)
    alpha = (0.5 * (-y2 * x3 + y * (-x2 + x3) + x * (y2 - y3) + x2 * y3)) / area_triangle
    beta = (0.5 * (y1 * x3 - x1 * y3 + x * (-y1 + y3) + x1 * y)) / area_triangle
    gamma = 1.0 - alpha - beta

    return np.array([alpha, beta, gamma])

def point_inside_triangle(point, triangle_vertices):
    """
    Check if a 2D point is contained on or in a triangle.

    Args:
        point (ndarray): 2D point as a 2-element array [x, y].
        triangle_vertices (ndarray): Triangle vertices as a 3x2 array [[x1, y1], [x2, y2], [x3, y3]].

    Returns:
        bool: True if the point is inside or on the triangle, False otherwise.
    """

    [alpha, beta, gamma] = barycentric_coordinates(point, triangle_vertices)

    # Check if the point is inside the triangle
    result = (0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1)

    #####################
    #
    result2 = point_inside_triangleX(point, triangle_vertices)
    log.debug(" result2 = %s", result2)
    #
    #####################

    barycentric = [alpha, beta, gamma]
    return [result, barycentric]


def which_face(point, face_indices, face_vertices):
    """
    Determine which face contains the point.

    Args:
        point (xyz): The input 2D points with shape (2,n).
        face_indices (ndarray): The indices of the possible faces.
        face_vertices (Trimesh): The vertices of the possible faces.

    Returns:
        index (int): The index of the possible face or None if the point intersects multiple faces.
        barycentric (ndarray): The 3 element array of barycentric coordinates of the point with respect to the triangle.
    """
    combined_results = [point_inside_triangle(point[:2], face_vertex[:, :2]) for face_vertex in face_vertices]
    results = [result[0] for result in combined_results]
    if np.sum(results) != 1:
        log.debug(" Unable to match point %s to face: %d matches <- %s", point, np.sum(results), face_indices)
        return None, None
    result_index = np.where(results)[0]
    barycentric = combined_results[result_index,1]
    return face_indices[result_index], barycentric
################################################



def test_uv_to_xyz_planar():
    val = build_planar_mesh(0.30, 0.60, 3, 3, 0, 0, 1, 0, 0, 0, 1.0)
    points_2d_in = np.array([[-0.15, -0.14], [-0.29, -0.10], [-0.19, +0.056]]).T  # Faces: 0, 1, 2
    test_faces = [0,1,4]

    curved_mesh = Trimesh(vertices=val.vertices, faces=val.faces)
    planar_uv_3d = curved_mesh.vertices
    planar_uv = planar_uv_3d[:, :2]

    points_out_3d, points_2d_out = uv_to_xyz(points_2d_in, planar_uv, curved_mesh)
    log.debug("points_2d_out: %s", points_2d_out)
    log.debug("points_out_3d: %s", points_out_3d)

def test_which_face_planar():
    plane = build_planar_mesh(0.30, 0.60, 3, 3, 0, 0, 1, 0, 0, 0, 0.0)
    planar_mesh = Trimesh(vertices=plane.vertices, faces=plane.faces)
    proximity = ProximityQuery(planar_mesh)

    point = plane.vertices[10] # + [0.02, 0.01, 0.0]

    distance, vertex_id = proximity.vertex(point)
    possible_triangles = planar_mesh.vertex_faces[vertex_id]
    refined_triangles = possible_triangles[np.where(possible_triangles >= 0)]

    possible_faces = plane.faces[refined_triangles]
    faces = which_face(point, refined_triangles, plane.vertices[possible_faces])
    log.debug("faces for %s: %s", point, faces)

def test_get_face_planar():
    points_2d_in = np.array([[-0.15, -0.14], [-0.29, -0.10], [-0.19, +0.056], [-1, -1]])  # Faces: 0, 1, 2
    test_faces = [0,1,4, None]
    plane = build_planar_mesh(0.30, 0.60, 3, 3, 0, 0, 1, 0, 0, 0, 0.0)
    planar_mesh = Trimesh(vertices=plane.vertices, faces=plane.faces)
    points_3d_in = np.zeros((len(points_2d_in),3))
    points_3d_in[:,:2] = points_2d_in
    for index, point in enumerate(points_3d_in):
        face = get_face(point, planar_mesh)
        # log.debug("face for %s: %s", point, face)
        assert face == test_faces[index]


if __name__ == "__main__":
    # Logging
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    test_get_face_planar()
    #test_which_face_planar()
    #test_uv_to_xyz_planar()
