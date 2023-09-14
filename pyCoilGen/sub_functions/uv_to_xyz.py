import numpy.linalg as la
import numpy as np

from trimesh import Trimesh
from trimesh.proximity import ProximityQuery

# Local imports
from .constants import get_level, DEBUG_VERBOSE

# Logging
import logging
log = logging.getLogger(__name__)


def uv_to_xyz_obsolete(points_in_2d_in: np.ndarray, planary_uv: np.ndarray, curved_mesh: Trimesh, num_attempts=1000):
    """
    Convert 2D surface coordinates to 3D xyz coordinates of the 3D coil surface.

    Args:
        points_in_2d (ndarray): The input 2D points with shape (2,n).
        planary_uv (ndarray): Mesh UV array (num vertices,2).
        curved_mesh (Trimesh): A curved mesh instance.
        num_attempts (int) : If the point is not on the mesh, how many times to search nearby points.

    Returns:
        points_out_3d (ndarray): The 3D xyz coordinates of the points with shape (3,n).
        points_in_2d (ndarray): The updated 2D points after removing points that could not be assigned to a triangle, with shape (2,n).
    """
    # NOTE: MATLAB coords
    # Use Trimesh and helpers
    planary_uv_3d = np.empty((planary_uv.shape[0], 3))
    planary_uv_3d[:, 0:2] = planary_uv
    planary_uv_3d[:, 2] = 0
    planary_mesh = Trimesh(faces=curved_mesh.faces, vertices=planary_uv_3d)
    planar_vertices = planary_mesh.vertices.view(np.ndarray)
    planar_faces = planary_mesh.faces.view(np.ndarray)
    mean_pos = np.mean(planar_vertices, axis=0)
    diameters = np.linalg.norm(planar_vertices - mean_pos, axis=1)
    avg_mesh_diameter = np.mean(diameters)

    # Create 3D array from 2D array [x,y] -> [x,y,0]
    points_in_3d = np.vstack((points_in_2d_in, np.zeros(points_in_2d_in.shape[1])))  # Create 3D equivalent (2,n -> 3,n)
    points_in_3d = points_in_3d.T  # n,3

    proximity = ProximityQuery(planary_mesh)
    points_out_3d = np.zeros((points_in_2d_in.shape[1], 3))
    num_deleted_points = 0
    for point_ind in range(points_in_3d.shape[0]):
        point = points_in_3d[point_ind - num_deleted_points]
        # Find the target triangle and barycentric coordinates of the point on the planar mesh
        target_triangle, barycentric = get_target_triangle_obsolete(point, planary_mesh, proximity)

        attempts = 0
        np.random.seed(3)  # Setting the seed to improve testing robustness
        while target_triangle is None:
            # If the point is not directly on a triangle, perturb the point slightly and try again
            rand = (0.5 - np.random.rand(2))
            perturbed_point = point + avg_mesh_diameter * np.array([rand[0], rand[1], 0.0]) / 1000
            target_triangle, barycentric = get_target_triangle_obsolete(perturbed_point, planary_mesh, proximity)
            attempts += 1
            if attempts > num_attempts:
                log.warning('point %s at index %d can not be assigned to any triangle.', point, point_ind)
                break

        if target_triangle is not None:
            if attempts > 0:
                point = perturbed_point
            # Convert the 2D barycentric coordinates to 3D Cartesian coordinates
            face_vertices = planar_vertices[planar_faces[target_triangle]]
            face_vertices_3d = curved_mesh.vertices[curved_mesh.faces[target_triangle]]
            points_out_3d[point_ind - num_deleted_points, :] = barycentric_to_cartesian(barycentric, face_vertices_3d)
        else:
            # Remove the point if it cannot be assigned to a triangle
            points_in_3d = np.delete(points_in_3d, point_ind - num_deleted_points, axis=0)
            points_out_3d = np.delete(points_out_3d, point_ind - num_deleted_points, axis=0)
            num_deleted_points += 1

    # Recover 2D array from 3D array [x,y,0] -> [x,y]
    points_in_2d = points_in_3d[:, :2]

    return points_out_3d.T, points_in_2d.T


def point_inside_triangle(point, triangle_vertices):
    """
    Check if a 2D point is contained on or in a triangle.

    Args:
        point (ndarray): 2D point as a 2-element array [x, y].
        triangle_vertices (ndarray): Triangle vertices as a 3x2 array [[x1, y1], [x2, y2], [x3, y3]].

    Returns:
        bool: True if the point is inside or on the triangle, False otherwise.
        barycentric (list): The barycentric coordinates of the point as a 1x3 array [alpha, beta, gamma].
    """

    [alpha, beta, gamma] = barycentric_coordinates(point, triangle_vertices)

    # Check if the point is inside the triangle
    close_enough = 1e-10
    lower = 0.0 - close_enough
    upper = 1.0 + close_enough
    return [(lower <= alpha <= upper) and (lower <= beta <= upper) and (lower <= gamma <= upper), [alpha, beta, gamma]]


def which_face(point, face_indices, face_vertices):
    """
    Determine which of the provided faces contains a given point.

    Args:
        point (xyz): The input 2D points with shape (3,).
        face_indices (ndarray): The indices of the possible faces (n,3).
        face_vertices (ndarray): The vertices of the possible faces (n,3,3).

    Returns:
        index (int): The index of the possible face or None if the point intersects multiple faces.
        barycentric (ndarray): The barycentric coordinates of the point as a 1x3 array [alpha, beta, gamma].
    """
    combined_results = [point_inside_triangle(point[:2], face_vertex[:, :2]) for face_vertex in face_vertices]
    results = [sublist[0] for sublist in combined_results]
    if np.sum(results) != 1:
        if get_level() > DEBUG_VERBOSE:
            log.debug(" Unable to match point %s to face: %d matches <- %s", point, np.sum(results), face_indices)
        return None, None
    result_index = np.where(results)[0][0]
    coords = [sublist[1] for sublist in combined_results]
    return face_indices[result_index], combined_results[result_index][1]


def pointLocation(point_2D: np.ndarray, face_indices: np.ndarray, mesh_vertices: np.ndarray):
    """
    Determine which of the provided faces contains a given point.

    If the point lies on an edge, i.e. multiple faces, the largest face index is returned.

    Args:
        point (xy): The input 2D points with shape (2,n).
        face_indices (ndarray): The indices of the possible faces (n,3).
        mesh_vertices (ndarray): The vertices of the mesh (m,3).

    Returns:
        index (int): The index of the possible face or None if the point intersects multiple faces.
        barycentric (list): The barycentric coordinates of the point as a 1x3 array [alpha, beta, gamma].
    """
    for index in range(len(face_indices)-1, -1, -1):
        face = face_indices[index]
        triangle_vertices = mesh_vertices[face]
        found, barycentric = point_inside_triangle(point_2D, triangle_vertices)
        if found:
            return index, barycentric
    return None, None


def get_target_triangle_def_obsolete(point, planary_mesh: Trimesh):
    """
    Get the face that contains the given point.

    This function is a helper that creates the required ProximityQuery using the given mesh.

    Args:
        point (ndarray): The input 2D point (x,y,0)
        planary_mesh (Trimesh): The 3D Trimesh representation of the 2D mesh (x,y,0)

    Returns:
        face (int): The index of the triangle that contains the point else None.
        barycentric (ndarray): The barycentric coordinates of the point as a 1x3 array [alpha, beta, gamma].

    """
    return get_target_triangle_obsolete(point, planary_mesh, ProximityQuery(planary_mesh))


def get_target_triangle_obsolete(point, planary_mesh: Trimesh, proximity: ProximityQuery):
    """
    Get the face that contains the given point using the provided ProximityQuery.

    Args:
        point (ndarray): The input 2D point (x,y,0)
        planary_mesh (Trimesh): The 3D Trimesh representation of the 2D mesh (x,y,0)
        proximity (ProximityQuery): The ProximityQuery helper to use.

    Returns:
        face (int): The index of the triangle that contains the point else None.
        barycentric (ndarray): The barycentric coordinates of the point as a 1x3 array [alpha, beta, gamma].

    """
    distance, vertex_id = proximity.vertex(point)
    possible_triangles = planary_mesh.vertex_faces[vertex_id]
    refined_triangles = possible_triangles[np.where(possible_triangles >= 0)]
    if len(refined_triangles) > 0:
        planar_vertices = planary_mesh.vertices.view(np.ndarray)
        face_indices = planary_mesh.faces[refined_triangles]
        target_triangle, barycentric = which_face(point, refined_triangles, planar_vertices[face_indices])
        return target_triangle, barycentric
    log.debug("Unable to find any face for point %s", point)
    return None, None


def barycentric_coords(point, vertices):
    T = (np.array(vertices[:-1])-vertices[-1]).T
    v = np.dot(la.inv(T), np.array(point)-vertices[-1])
    # v.resize(len(vertices)
    # v[-1] = 1-v.sum()
    v = np.append(v, 1-v.sum())
    return v


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
    triangle_area = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / triangle_area
    beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / triangle_area
    gamma = 1 - alpha - beta

    v = barycentric_coords(point, triangle_vertices)

    return v  # [alpha, beta, gamma]


def barycentric_to_cartesian(bary_coords, triangle_vertices):
    """
    Convert Barycentric coordinates to 3D Cartesian coordinates.

    Args:
        bary_coords (ndarray): 3-element array representing the Barycentric coordinates.
        triangle_vertices (ndarray): 3x3 array representing the 3D coordinates of the triangle vertices.

    Returns:
        ndarray: 3-element array representing the 3D Cartesian coordinates.
    """
    # Calculate the Cartesian coordinates using the Barycentric coordinates
    cartesian_coords = np.dot(bary_coords, triangle_vertices)
    return cartesian_coords
