import numpy as np
from trimesh.points import point_plane_distance
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
        points_out_3d (ndarray): The 3D xyz coordinates of the points with shape (3,n).
        points_in_2d (ndarray): The updated 2D points after removing points that could not be assigned to a triangle, with shape (2,n).
    """
    # Use Trimesh and helpers
    planary_uv_3d = np.empty((planary_uv.shape[0],3))
    planary_uv_3d[:,0:2] = planary_uv
    planary_uv_3d[:,2] = 0
    planary_mesh = Trimesh(faces=curved_mesh.faces, vertices=planary_uv_3d)
    planar_vertices = planary_mesh.vertices
    mean_pos = np.mean(planar_vertices, axis=0)
    diameters = np.linalg.norm(planar_vertices - mean_pos, axis=1)
    avg_mesh_diameter = np.mean(diameters)

    # Create 3D array from 2D array [x,y] -> [x,y,0]
    points_in_3d = np.vstack((points_in_2d_in, np.zeros(points_in_2d_in.shape[1])))  # Create 3D equivalent (2,n -> 3,n)
    points_in_3d = points_in_3d.T # n,3

    points_out_3d = np.zeros((points_in_2d_in.shape[1], 3))
    num_deleted_points = 0
    for point_ind in range(points_in_3d.shape[0]):
        point = points_in_3d[point_ind - num_deleted_points]
        ############################
        # DEBUG
        # DEBUG:sub_functions.uv_to_xyz: Unable to match point [-1.61478303  0.02161017  0.        ] to face: 2 matches
        # DEBUG:sub_functions.uv_to_xyz: point: 0 at [-1.61478303  0.02161017  0.        ], possible_triangles: [ 67 160]        
        if np.allclose(point, [-1.61478303,  0.02161017,  0.]):
            log.debug(" Here! this point!")
            # target_triangles = [373]
        #
        ############################
        # Find the target triangle and barycentric coordinates of the point on the planar mesh
        closest, bary_centric_coord, target_triangles = planary_mesh.nearest.on_surface([point])
        if (len(target_triangles) == 1):
            target_triangle = target_triangles[0]
        else:
            target_triangle = None

        tri_inds = 0
        while target_triangle is None:
            # If the point is not directly on a triangle, perturb the point slightly and try again
            rand = (0.5 - np.random.rand(2))
            perturbed_point = point + avg_mesh_diameter * np.array([ rand[0], rand[1], 0.0]) / 100
            closest, bary_centric_coord, target_triangles = planary_mesh.nearest.on_surface([perturbed_point])
            if (len(target_triangles) == 1):
                target_triangle = target_triangles[0]
            else:
                target_triangle = None

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

    # Recover 2D array from 3D array [x,y,0] -> [x,y]
    points_in_2d = points_in_3d[:,:2]

    return points_out_3d.T, points_in_2d.T