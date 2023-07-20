import numpy as np
from trimesh.points import point_plane_distance
from trimesh.proximity import ProximityQuery
from trimesh import Trimesh

# Logging
import logging
log = logging.getLogger(__name__)


def uv_to_xyz(points_in_2d_in: np.ndarray, planary_uv : np.ndarray, curved_mesh: Trimesh):
    """
    Convert 2D surface coordinates to 3D xyz coordinates of the 3D coil surface.

    Args:
        points_in_2d (ndarray): The input 2D points with shape (2,n).
        planary_uv (ndarray): Mesh UV array (num vertices,2).
        curved_mesh (Trimesh): A curved mesh instance.

    Returns:
        ndarray: The 3D xyz coordinates of the points with shape (N, 3).
        ndarray: The updated 2D points after removing points that could not be assigned to a triangle, with shape (M, 2).
    """
    # Use Trimesh and helpers
    planary_uv_3d = np.empty((planary_uv.shape[0],3))
    planary_uv_3d[:,0:2] = planary_uv
    planary_uv_3d[:,2] = 0
    planary_mesh = Trimesh(faces=curved_mesh.faces, vertices=planary_uv_3d)
    planar_vertices = planary_mesh.vertices
    # TODO: Revert this
    # mean_pos = np.mean(planar_vertices, axis=0)
    # diameters = np.linalg.norm(planar_vertices - mean_pos, axis=1)
    avg_mesh_diameter = 20.0# np.mean(diameters)

    points_out_3d = np.zeros((points_in_2d_in.shape[1], 3))

    points_in_3d = np.vstack((points_in_2d_in, np.zeros(points_in_2d_in.shape[1])))  # Create 3D equivalent (2,n -> 3,n)
    points_in_3d = points_in_3d.T # n,3
    num_deleted_points = 0
    proximity = ProximityQuery(planary_mesh)
    for point_ind in range(points_in_3d.shape[0]):
        point = points_in_3d[point_ind - num_deleted_points]
        distance, vertex_id = proximity.vertex(point)
        possible_triangles = planary_mesh.face_adjacency[vertex_id]
        target_triangle = which_face(point, possible_triangles, planar_vertices[planary_mesh.faces[possible_triangles]])

        tri_inds = 0
        while target_triangle is None:
            # If the point is not directly on a triangle, perturb the point slightly and try again
            rand = (0.5 - np.random.rand(2))
            old_point = point
            point = point + np.array([avg_mesh_diameter * rand[0], avg_mesh_diameter * rand[1], 0.0]) / 100
            distance, vertex_id = proximity.vertex(point)
            possible_triangles = planary_mesh.face_adjacency[vertex_id]
            target_triangle = which_face(point, possible_triangles, planar_vertices[planary_mesh.faces[possible_triangles]])

            tri_inds += 1

            if tri_inds > 1000:
                print('Warning: Points cannot be assigned to a triangle')
                log.warning('point %s at index %d can not be assigned to any triangle.', point, point_ind)
                break

        if target_triangle is not None:
            # Convert the 2D barycentric coordinates to 3D Cartesian coordinates
            points_out_3d[point_ind - num_deleted_points, :] = point_plane_distance(
                curved_mesh.vertices[curved_mesh.faces[target_triangle]],
                curved_mesh.face_normals[target_triangle],
                points_in_3d[point_ind - num_deleted_points, :]
            )
        else:
            # Remove the point if it cannot be assigned to a triangle
            points_in_3d = np.delete(points_in_3d, point_ind - num_deleted_points, axis=0)
            points_out_3d = np.delete(points_out_3d, point_ind - num_deleted_points, axis=0)
            num_deleted_points += 1

    return points_out_3d, points_in_3d.T


def point_inside_trigon(point, face):
    as_x = point[0] - face[0,0]
    as_y = point[1] - face[0,1]

    s_ab = (face[1,0] - face[0,0]) * as_y - (face[1,1] - face[0,1]) * as_x > 0

    if ((face[2,0] - face[0,0]) * as_y - (face[2,1] - face[0,1]) * as_x > 0 == s_ab):
        return False;
    if ((face[2,0] - face[1,0]) * (point[1] - face[1,1]) - (face[2,1] - face[1,1])*(point[0] - face[1,0]) > 0 != s_ab):
        return False
    return True

def which_face(point, face_indices, face_vertices):
    """
    Determine which face contains the point.

    Args:
        point (xyz): The input 2D points with shape (2,n).
        face_indices (ndarray): The indices of the possible faces.
        face_vertices (Trimesh): The vertices of the possible faces.
    
    Returns:
        index (int): The index of the possible face or None if the point intersects multiple faces.

    """
    results = [point_inside_trigon(point, face_vertex) for face_vertex in face_vertices]
    if np.sum(results) != 1:
        # log.debug(" Unable to match point %s to face: %d matches", point, np.sum(results))
        return None
    return face_indices[np.where(results)][0]