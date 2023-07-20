import numpy as np
from trimesh.points import point_plane_distance
from trimesh import Trimesh

# Logging
import logging
log = logging.getLogger(__name__)


def uv_to_xyz(points_in_2d_in : np.ndarray, planary_mesh : Trimesh, curved_mesh : Trimesh):
    """
    Convert 2D surface coordinates to 3D xyz coordinates of the 3D coil surface.

    Args:
        points_in_2d (ndarray): The input 2D points with shape (2, N).
        planary_mesh (Trimesh): A planar mesh instance.
        curved_mesh (Trimesh): A curved mesh instance.

    Returns:
        ndarray: The 3D xyz coordinates of the points with shape (N, 3).
        ndarray: The updated 2D points after removing points that could not be assigned to a triangle, with shape (M, 2).
    """
    points_in_2d = points_in_2d_in.T
    num_deleted_points = 0
    points_out_3d = np.zeros((points_in_2d.shape[1], 3))
    planar_vertices = planary_mesh.vertices
    #operands could not be broadcast together with shapes (264,2) (264,) 
    mean_pos = np.mean(planar_vertices, axis=0)
    avg_mesh_diameter = np.linalg.norm(planar_vertices - mean_pos)

    for point_ind in range(points_in_2d.shape[0]):
        # Find the target triangle and barycentric coordinates of the point on the planar mesh
        # Exception : points must be (n,3)!
        target_triangle, bary_centric_coord = planary_mesh.nearest.on_surface(
            [points_in_2d[point_ind - num_deleted_points, :]])

        tri_inds = 0
        while target_triangle is None:
            # If the point is not directly on a triangle, perturb the point slightly and try again
            perturbed_point = points_in_2d[point_ind - num_deleted_points,:] + \
                avg_mesh_diameter * (0.5 - np.random.rand(2)) / 1000
            target_triangle, bary_centric_coord = planary_mesh.nearest.on_surface([perturbed_point])
            tri_inds += 1

            if tri_inds > 1000:
                print('Warning: Points cannot be assigned to a triangle')
                break

        if target_triangle is not None:
            # Convert the 2D barycentric coordinates to 3D Cartesian coordinates
            points_out_3d[point_ind - num_deleted_points,:] = point_plane_distance(
                curved_mesh.vertices[curved_mesh.faces[target_triangle]],
                curved_mesh.face_normals[target_triangle],
                points_in_2d[point_ind - num_deleted_points,:]
            )
        else:
            # Remove the point if it cannot be assigned to a triangle
            points_in_2d = np.delete(points_in_2d, point_ind - num_deleted_points, axis=0)
            points_out_3d = np.delete(points_out_3d, point_ind - num_deleted_points, axis=0)
            num_deleted_points += 1

    return points_out_3d, points_in_2d.T
