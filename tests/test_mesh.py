import numpy as np

from sub_functions.data_structures import Mesh


def test_get_face_index1():
    vertices = np.asarray([[-1., -1.], [+1., -1.], [0., 0.], [-1., 1.], [1., 1.]])
    faces = np.asarray([[0, 1, 2], [2, 1, 4], [4, 3, 2], [2, 3, 0]])
    mesh = Mesh(vertices=vertices, faces=faces)

    point = [-1., 0.]
    ############################################
    # Function under test
    index, barycentric = mesh.get_face_index(point)
    ############################################
    assert index == 3

    y = -0.4
    point2 = [[-1.0, y], [-0.4, y], [0, y], [0.4, y], [1.0, y]]
    indices = [3, 3, 0, 1, 1]

    for index, point in enumerate(point2):
        ############################################
        # Function under test
        face_index, barycentric = mesh.get_face_index(point)
        ############################################
        assert face_index == indices[index]

    y = 0.4
    point2 = [[-1.0, y], [-0.4, y], [0, y], [0.4, y], [1.0, y]]
    indices = [3, 3, 2, 2, 1]

    for index, point in enumerate(point2):
        ############################################
        # Function under test
        face_index, barycentric = mesh.get_face_index(point)
        ############################################
        assert face_index == indices[index]

    point = [0., 0.]
    ############################################
    # Function under test
    index, barycentric = mesh.get_face_index(point)
    ############################################
    assert index == 3


# There is something about the wire_mesh2D.get_face_index and pointLocation implementation that returns different
# results compared to MATLAB. This test proves it!!
def test_get_face_index2():
    # z = 0.0
    # vertices = np.asarray([[-1., 0., z], [0.0, -5.0, z], [11.0, 0.0, z], [0.0, 5.0, z]])
    vertices = np.asarray([[-1., 0.], [0.0, -5.0], [11.0, 0.0], [0.0, 5.0]])
    faces = np.asarray([[0, 1, 3], [3, 1, 2]])
    mesh = Mesh(vertices=vertices, faces=faces)

    y = 0
    points = [[-2.0, y], [-1.0, y], [0, y]]
    indices = [-1, 0, 1]

    for index, point in enumerate(points):
        ############################################
        # Function under test
        face_index, barycentric = mesh.get_face_index(point, try_harder=True)
        ############################################
        if face_index != indices[index]:
            print("Fails at index", index)
        assert face_index == indices[index]

def test_uv_to_xyz_planar():
    from sub_functions.build_planar_mesh import build_planar_mesh
    val = build_planar_mesh(0.30, 0.60, 3, 3, 0, 0, 1, 0, 0, 0, 1.0)
    points_2d_in = np.array([[-0.15, -0.14], [-0.29, -0.10], [-0.19, +0.056]]).T  # Faces: 0, 1, 2

    curved_mesh = Mesh(vertices=val.vertices, faces=val.faces)
    planar_uv_3d = curved_mesh.get_vertices()
    planar_uv = planar_uv_3d[:, :2]

    # Test code
    curved_mesh.v = curved_mesh.get_vertices()
    curved_mesh.f = curved_mesh.get_faces()
    points_out_3d, points_2d_out = curved_mesh.uv_to_xyz(points_2d_in, planar_uv)

    assert points_out_3d.shape == (3,3)
    assert points_2d_out.shape == (2,3)

    assert np.isclose(points_out_3d[0], points_2d_out[0]).all()
    assert np.isclose(points_out_3d[1], points_2d_out[1]).all()
    assert np.isclose(points_out_3d[2], [1., 1., 1.]).all()

    # Verify that a point not on the mesh is not assigned to a face and removed from the input data.
    points_2d_in = np.array([[-0.15, -0.14], [-0.29, -0.10], [-0.19, +0.056], [-1, -1]]).T  # Faces: 0, 1, 2, None

    # Test code
    points_out_3d, points_2d_out = curved_mesh.uv_to_xyz(points_2d_in, planar_uv, 3)
    assert points_out_3d.shape == (3,3)
    assert points_2d_out.shape == (2,3)

    # Verify that a point close to the mesh can be placed onto the mesh (Random!! Can fail!)
    points_2d_in = np.array([[-0.3001, -0.1]]).T  # Faces: 0, 1, 2, None

    # Test code
    points_out_3d, points_2d_out = curved_mesh.uv_to_xyz(points_2d_in, planar_uv, 100)
    assert points_out_3d.shape == (3,1)
    assert points_2d_out.shape == (2,1)