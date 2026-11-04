import numpy as np
import pytest

# Test support
from pytest import approx

# Code under test
from pyCoilGen.mesh_factory.build_circular_biplanar_mesh import build_circular_biplanar_mesh, simple_circular_mesh
from pyCoilGen.mesh_factory import load_mesh_factory_plugins


def _face_normals(vertices, faces):
    v1, v2, v3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return np.cross(v2 - v1, v3 - v1)


def test_simple_circular_mesh_counts():
    nr, nt = 5, 12
    vertices, faces = simple_circular_mesh(0.5, nr, nt)

    assert vertices.shape == (1 + nr * nt, 3)
    # Fan of nt triangles, plus two triangles per angular division for each of the remaining rings
    assert faces.shape == (nt + 2 * nt * (nr - 1), 3)
    # All vertices lie on z=0 and within the radius
    assert np.all(vertices[:, 2] == 0.0)
    assert np.max(np.linalg.norm(vertices, axis=1)) == approx(0.5)
    # Every vertex is used
    assert set(np.unique(faces)) == set(range(vertices.shape[0]))


def test_simple_circular_mesh_area_and_winding():
    vertices, faces = simple_circular_mesh(1.0, 20, 90)
    normals = _face_normals(vertices, faces)
    assert np.all(normals[:, 2] > 0)
    # Polygon approximation of a unit circle
    area = 0.5 * np.sum(normals[:, 2])
    assert area == approx(0.5 * 90 * np.sin(2 * np.pi / 90) * 1.0**2, rel=1e-9)

    _, reversed_faces = simple_circular_mesh(1.0, 20, 90, normal_winding_direction=False)
    assert np.all(_face_normals(vertices, reversed_faces)[:, 2] < 0)


@pytest.mark.parametrize("radius, nr, nt", [(0.0, 5, 12), (-1.0, 5, 12), (1.0, 0, 12), (1.0, 5, 2)])
def test_simple_circular_mesh_invalid(radius, nr, nt):
    with pytest.raises(ValueError):
        simple_circular_mesh(radius, nr, nt)


def test_build_circular_biplanar_mesh_basic():
    radius, nr, nt, distance = 0.07, 10, 36, 0.0735
    mesh = build_circular_biplanar_mesh(radius, nr, nt, 0, 0, 1, 0, 0, 0, distance)

    assert mesh.vertices.shape == (2 * (1 + nr * nt), 3)
    assert np.min(mesh.vertices[:, 2]) == approx(-distance / 2.0)
    assert np.max(mesh.vertices[:, 2]) == approx(distance / 2.0)
    assert np.max(np.abs(mesh.vertices[:, 0])) == approx(radius)
    assert np.allclose(mesh.normal, [0, 0, 1])

    # Normals point away from the other plate
    normals = _face_normals(mesh.vertices, mesh.faces)
    lower = mesh.vertices[mesh.faces[:, 0], 2] < 0
    assert np.all(normals[lower, 2] < 0)
    assert np.all(normals[~lower, 2] > 0)


def test_build_circular_biplanar_mesh_x_axis_and_offset():
    radius, nr, nt, distance = 0.07, 6, 24, 0.1
    mesh = build_circular_biplanar_mesh(radius, nr, nt, 1, 0, 0, 0.1, 0.2, 0.3, distance)

    # The plates are separated along x, centred on the centre position
    assert np.min(mesh.vertices[:, 0]) == approx(0.1 - distance / 2.0)
    assert np.max(mesh.vertices[:, 0]) == approx(0.1 + distance / 2.0)
    assert np.mean(mesh.vertices[:, 1]) == approx(0.2)
    assert np.mean(mesh.vertices[:, 2]) == approx(0.3)


def test_circular_biplanar_mesh_plugin():
    plugins = [p for p in load_mesh_factory_plugins() if getattr(p, 'get_name', lambda: None)() == 'create circular bi-planar mesh']
    assert len(plugins) == 1
    plugin = plugins[0]

    parameter_name, _ = plugin.get_parameters()[0]
    input_args = type('Args', (), {parameter_name: [0.07, 8, 24, 0, 0, 1, 0, 0, 0, 0.0735]})()
    creator = getattr(plugin, 'create_circular_bi_planar_mesh')
    coil_mesh = creator(input_args)

    assert coil_mesh.get_vertices().shape == (2 * (1 + 8 * 24), 3)
