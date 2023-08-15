# Hack code
from trimesh import Trimesh
from trimesh.proximity import ProximityQuery
import numpy as np


# Set up paths: Add the project root directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Test support
from sub_functions.build_planar_mesh import build_planar_mesh
from sub_functions.data_structures import Mesh
# Code under test
<<<<<<< HEAD
from sub_functions.uv_to_xyz import uv_to_xyz, which_face, get_target_triangle, pointLocation
=======
from sub_functions.uv_to_xyz import uv_to_xyz, which_face, get_target_triangle
>>>>>>> c89b1f1cff49a9685e407f22c881124cab1b1724

def test_uv_to_xyz_planar():
    val = build_planar_mesh(0.30, 0.60, 3, 3, 0, 0, 1, 0, 0, 0, 1.0)
    points_2d_in = np.array([[-0.15, -0.14], [-0.29, -0.10], [-0.19, +0.056]]).T  # Faces: 0, 1, 2

    curved_mesh = Trimesh(vertices=val.vertices, faces=val.faces)
    planar_uv_3d = curved_mesh.vertices
    planar_uv = planar_uv_3d[:, :2]

    # Test code
    points_out_3d, points_2d_out = uv_to_xyz(points_2d_in, planar_uv, curved_mesh)

    assert points_out_3d.shape == (3,3)
    assert points_2d_out.shape == (2,3)

    assert np.isclose(points_out_3d[0], points_2d_out[0]).all()
    assert np.isclose(points_out_3d[1], points_2d_out[1]).all()
    assert np.isclose(points_out_3d[2], [1., 1., 1.]).all()

    # Verify that a point not on the mesh is not assigned to a face and removed from the input data.
    points_2d_in = np.array([[-0.15, -0.14], [-0.29, -0.10], [-0.19, +0.056], [-1, -1]]).T  # Faces: 0, 1, 2, None

    # Test code
    points_out_3d, points_2d_out = uv_to_xyz(points_2d_in, planar_uv, curved_mesh, 3)
    assert points_out_3d.shape == (3,3)
    assert points_2d_out.shape == (2,3)

    # Verify that a point close to the mesh can be placed onto the mesh (Random!! Can fail!)
    points_2d_in = np.array([[-0.3001, -0.1]]).T  # Faces: 0, 1, 2, None

    # Test code
    points_out_3d, points_2d_out = uv_to_xyz(points_2d_in, planar_uv, curved_mesh, 100)
    assert points_out_3d.shape == (3,1)
    assert points_2d_out.shape == (2,1)


def test_get_target_triangle():
    points_2d_in = np.array([[-0.15, -0.14], [-0.29, -0.10], [-0.19, +0.056], [-1, -1]])  # Faces: 0, 1, 2
    test_faces = [0,1,4, None]
    plane = build_planar_mesh(0.30, 0.60, 3, 3, 0, 0, 1, 0, 0, 0, 0.0)
    planar_mesh = Trimesh(vertices=plane.vertices, faces=plane.faces)
    proximity = ProximityQuery(planar_mesh)
    points_3d_in = np.zeros((len(points_2d_in),3))
    points_3d_in[:,:2] = points_2d_in
    for index, point in enumerate(points_3d_in):
        face, barycentric = get_target_triangle(point, planar_mesh, proximity)
        # log.debug("face for %s: %s", point, face)
        assert face == test_faces[index]

<<<<<<< HEAD
def test_pointLocation():
    mesh_faces = np.array([[0, 1, 2], [2, 3, 0], [3, 2, 4]])
    mesh_uv = np.array([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4], [0.1, 0.6]])

    face_index, barycentric = pointLocation([0.25, 0.25], mesh_faces, mesh_uv)
    assert face_index == 0

    face_index, barycentric = pointLocation([0.0, 0.0], mesh_faces, mesh_uv)
    assert face_index == None

    face_index, barycentric = pointLocation([0.2, 0.45], mesh_faces, mesh_uv)
    assert face_index == 2

    # One of the vertices, return the highest face that has that vertex
    face_index, barycentric = pointLocation(mesh_uv[2], mesh_faces, mesh_uv)
    assert face_index == 2
    assert barycentric == [0.0, 1.0, 0.0] #100% described by the 2nd co-ordinate

=======
>>>>>>> c89b1f1cff49a9685e407f22c881124cab1b1724

if __name__ == "__main__":
    # Logging
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    test_get_target_triangle()
    test_uv_to_xyz_planar()
<<<<<<< HEAD
    test_pointLocation()
=======
>>>>>>> c89b1f1cff49a9685e407f22c881124cab1b1724
