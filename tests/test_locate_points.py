"""
Compare the vectorised point location (`Mesh.get_face_indices`, `Mesh.uv_to_xyz`) with the original implementations.

The reference functions below are the original code, which searched the faces of each point separately, in Python.
They are kept as the specification that the fast implementation must reproduce.
"""
# Libraries
import numpy as np
import pytest

# Local imports
from pyCoilGen.mesh_factory.build_planar_mesh import build_planar_mesh
from pyCoilGen.sub_functions.data_structures import Mesh
from pyCoilGen.sub_functions.uv_to_xyz import pointLocation, barycentric_to_cartesian
import logging
log = logging.getLogger(__name__)


def reference_get_face_index(self, vertex: np.ndarray, try_harder=True):
    """
    Retrieve the index of the face which contains the provided point.

    If the vertex is on multiple faces, e.g. is an edge, the highest face index is returned.

    Args:
        vertex (ndarray): The point to search for
        try_harder (bool): Whether to also include 2nd order face vertices in the search.

    Returns:
        index (int): The face index or -1 if the point is not within the mesh.
        barycentric (ndarray): The 3 barycentric co-ordinates if the point is within the mesh, else None
    """
    faces = self.f
    vertices = self.v
    diffs = np.abs(vertices - vertex)
    diffs_norm = np.linalg.norm(diffs, axis=1)
    # Find the closest vertices
    min_index = np.argmin(diffs_norm)

    # For each possible vertex, find the faces that reference it.
    possible_face_matches = np.any(faces == min_index, axis=1)

    # Get the indices of the rows that contain the target_face_index
    possible_face_indices = np.where(possible_face_matches)[0]

    # Get the corresponding faces
    faces_to_try = faces[possible_face_indices]

    if try_harder:
        # Add all 2nd degree faces (i.e. faces connected to every vertex)
        more_face_indices = possible_face_indices.tolist()
        for vertex_index in faces_to_try.flatten():
            p2 = np.any(faces == vertex_index, axis=1)
            p2_inds = np.where(p2)[0]
            more_face_indices += p2_inds.tolist()

        possible_face_indices = np.unique(more_face_indices)

        # Get the faces that match these vertices
        faces_to_try = faces[possible_face_indices]

    face_index, barycentric = pointLocation(vertex, faces_to_try, vertices)
    if face_index is not None:
        return possible_face_indices[face_index], barycentric

    return -1, None

def reference_uv_to_xyz(self, points_in_2d_in: np.ndarray, planary_uv: np.ndarray, num_attempts=1000):
    """
    Convert 2D surface coordinates to 3D xyz coordinates of the 3D coil surface.

    Args:
        points_in_2d (ndarray): The input 2D points with shape (2,n).
        planary_uv (ndarray): Mesh UV array (num vertices,2).
        num_attempts (int) : If the point is not on the mesh, how many times to search nearby points.

    Returns:
        points_out_3d (ndarray): The 3D xyz coordinates of the points with shape (3,n).
        points_out_2d (ndarray): The updated 2D points after removing points that could not be assigned to a triangle, with shape (2,n).
    """
    # NOTE: MATLAB coords: points_in_2d, points_out_3d and points_out_2d
    curved_mesh_vertices = self.v
    curved_mesh_faces = self.f

    planary_mesh = Mesh(faces=curved_mesh_faces, vertices=planary_uv)
    planar_vertices = planary_mesh.trimesh_obj.vertices

    mean_pos = np.mean(planar_vertices, axis=0)
    diameters = np.linalg.norm(planar_vertices - mean_pos, axis=1)
    avg_mesh_diameter = np.mean(diameters)

    points_out_3d = np.zeros((points_in_2d_in.shape[1], 3))  # Python shape
    points_out_2d = points_in_2d_in.T.copy()  # Python shape
    num_deleted_points = 0
    for point_ind in range(points_out_2d.shape[0]):  # MATLAB sgape
        point = points_out_2d[point_ind - num_deleted_points]
        # Find the target triangle and barycentric coordinates of the point on the planar mesh
        # target_triangle, barycentric = get_target_triangle(point, planary_mesh, proximity)
        target_triangle, barycentric = reference_get_face_index(planary_mesh, point)

        attempts = 0
        np.random.seed(3)  # Setting the seed to improve testing robustness
        while target_triangle == -1:
            # If the point is not directly on a triangle, perturb the point slightly and try again
            rand = (0.5 - np.random.rand(2))
            perturbed_point = point + avg_mesh_diameter * np.array([rand[0], rand[1]]) / 1000
            # target_triangle, barycentric = get_target_triangle(perturbed_point, planary_mesh, proximity)
            target_triangle, barycentric = reference_get_face_index(planary_mesh, perturbed_point)
            attempts += 1
            if attempts > num_attempts:
                log.warning('point %s at index %d can not be assigned to any triangle.', point, point_ind)
                break

        if target_triangle != -1:
            if attempts > 0:
                point = perturbed_point
            # Convert the 2D barycentric coordinates to 3D Cartesian coordinates
            face_vertices_3d = curved_mesh_vertices[curved_mesh_faces[target_triangle]]
            points_out_3d[point_ind - num_deleted_points,
                          :] = barycentric_to_cartesian(barycentric, face_vertices_3d)
        else:
            # Remove the point if it cannot be assigned to a triangle
            points_out_2d = np.delete(points_out_2d, point_ind - num_deleted_points, axis=0)  # Python shape
            points_out_3d = np.delete(points_out_3d, point_ind - num_deleted_points, axis=0)  # Python shape
            num_deleted_points += 1

    return points_out_3d.T, points_out_2d.T  # Return as MATLAB shape



def _planar_mesh(kind):
    """A planar 2D mesh (regular, or with jittered vertices) and a curved 3D mesh with the same faces."""
    mesh3d = build_planar_mesh(0.30, 0.60, 8, 8, 0, 0, 1, 0, 0, 0, 1.0)
    uv = mesh3d.vertices[:, :2].copy()
    if kind == 'irregular':
        uv += np.random.default_rng(5).normal(scale=0.01, size=uv.shape)   # Distorted triangles, some overlap
    curved_vertices = np.column_stack((uv[:, 0], uv[:, 1], 0.1 * np.sin(8 * uv[:, 0])))
    return Mesh(faces=mesh3d.faces, vertices=uv), Mesh(faces=mesh3d.faces, vertices=curved_vertices)


def _test_points(mesh, rng):
    uv, faces = mesh.v, mesh.f
    corners = uv[faces]
    interior = np.einsum('nk,nkd->nd', rng.dirichlet([1, 1, 1], size=len(faces)), corners)
    edge_mid = (corners[:, 0] + corners[:, 1]) / 2
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    random_points = lo + rng.random((200, 2)) * (hi - lo) * 1.2 - (hi - lo) * 0.1   # Some outside
    return np.vstack((interior, uv, edge_mid, random_points, uv + 1e-12, lo - 1.0))


@pytest.mark.parametrize('kind', ['regular', 'irregular'])
@pytest.mark.parametrize('try_harder', [True, False])
def test_get_face_indices_matches_the_original(kind, try_harder):
    mesh, _ = _planar_mesh(kind)
    points = _test_points(mesh, np.random.default_rng(2))
    faces, barycentric = mesh.get_face_indices(points, try_harder)
    for i, point in enumerate(points):
        expected_face, expected_barycentric = reference_get_face_index(mesh, point, try_harder)
        assert faces[i] == expected_face, f"point {i} {point}"
        if expected_face != -1:
            np.testing.assert_allclose(barycentric[i], expected_barycentric, atol=1e-9)


def test_get_face_index_single_point_matches_the_original():
    mesh, _ = _planar_mesh('regular')
    for point in _test_points(mesh, np.random.default_rng(3))[:50]:
        face, bary = mesh.get_face_index(point)
        expected_face, expected_bary = reference_get_face_index(mesh, point)
        assert face == expected_face
        assert (bary is None) == (expected_bary is None)


@pytest.mark.parametrize('kind', ['regular', 'irregular'])
def test_uv_to_xyz_matches_the_original(kind):
    mesh2d, mesh3d = _planar_mesh(kind)
    curved = mesh3d
    points = _test_points(mesh2d, np.random.default_rng(4))
    # The points far outside are removed, the ones just outside are perturbed onto the mesh
    expected_3d, expected_2d = reference_uv_to_xyz(curved, points.T, mesh2d.v, 5)
    actual_3d, actual_2d = curved.uv_to_xyz(points.T, mesh2d.v, 5)
    assert actual_3d.shape == expected_3d.shape
    np.testing.assert_allclose(actual_2d, expected_2d)
    np.testing.assert_allclose(actual_3d, expected_3d, atol=1e-9)
