import numpy as np

from sub_functions.data_structures import Mesh


def test_get_face_index1():
    vertices = np.asarray([[-1., -1.], [+1., -1.], [0., 0.], [-1., 1.], [1., 1.]])
    faces = np.asarray([[0, 1, 2], [2, 1, 4], [4, 3, 2], [2, 3, 0]])
    mesh = Mesh(vertices=vertices, faces=faces)

    point = [-1., 0.]
    ############################################
    # Function under test
    index, possible_face_indices, faces_to_try = mesh.get_face_index(point)
    ############################################
    assert index == 3

    y = -0.4
    point2 = [[-1.0, y], [-0.4, y], [0, y], [0.4, y], [1.0, y]]
    indices = [3, 3, 0, 1, 1]

    for index, point in enumerate(point2):
        ############################################
        # Function under test
        face_index, possible_face_indices, faces_to_try = mesh.get_face_index(point)
        ############################################
        assert face_index == indices[index]

    y = 0.4
    point2 = [[-1.0, y], [-0.4, y], [0, y], [0.4, y], [1.0, y]]
    indices = [3, 3, 2, 2, 1]

    for index, point in enumerate(point2):
        ############################################
        # Function under test
        face_index, possible_face_indices, faces_to_try = mesh.get_face_index(point)
        ############################################
        assert face_index == indices[index]

    point = [0., 0.]
    ############################################
    # Function under test
    index, possible_face_indices, faces_to_try = mesh.get_face_index(point)
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
        face_index, possible_face_indices, faces_to_try = mesh.get_face_index(point, try_harder=True)
        ############################################
        if face_index != indices[index]:
            print("Fails at index", index)
        assert face_index == indices[index]
