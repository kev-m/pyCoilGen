import warnings

import numpy as np

from pyCoilGen.mesh_factory.build_planar_mesh import build_planar_mesh
from pyCoilGen.sub_functions.data_structures import Mesh
# Code under test
from pyCoilGen.sub_functions.mesh_parameterization_iterative import mesh_parameterization_iterative


def test_mesh_parameterization_iterative_of_a_curved_patch():
    # A curved, open surface: it is neither planar nor a cylinder, so it needs the iterative parameterization.
    # Regression test: with numpy 2.5, assigning the one-element array of the opposite vertex raised a ValueError.
    flat = build_planar_mesh(0.25, 0.4, 5, 6, 0, 0, 1, 0, 0, 0, 0)
    vertices = np.array(flat.vertices).copy()
    vertices[:, 2] = 0.8 * vertices[:, 0] ** 2

    mesh = Mesh(vertices=vertices, faces=np.array(flat.faces))
    mesh.v, mesh.f = mesh.get_vertices(), mesh.get_faces()
    mesh.unique_vert_inds = np.arange(len(mesh.v))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # scipy warns that spsolve prefers CSC or CSR
        result = mesh_parameterization_iterative(mesh)

    assert result.uv.shape == (len(vertices), 2)
    assert np.all(np.isfinite(result.uv))
    # Two boundary vertices are fixed at (0,0) and (1,0)
    assert np.min(result.uv[:, 0]) == 0.0 and np.isclose(np.max(result.uv[:, 0]), 1.0)
