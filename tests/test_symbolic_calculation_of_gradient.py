import numpy as np

from pyCoilGen.sub_functions.data_structures import DataStructure
# Code under test
from pyCoilGen.sub_functions.define_target_field import symbolic_calculation_of_gradient


def test_symbolic_calculation_of_gradient_is_evaluated_at_the_points():
    input_args = DataStructure(debug=1, field_shape_function='x + y**2 + x*z')
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (3, 10))
    dx, dy, dz = symbolic_calculation_of_gradient(input_args=input_args, target_points=points)

    np.testing.assert_allclose(dx, 1 + points[2])
    np.testing.assert_allclose(dy, 2 * points[1])
    np.testing.assert_allclose(dz, points[0])


def test_symbolic_calculation_of_gradient_expands_constants():
    input_args = DataStructure(debug=1, field_shape_function='x')
    dx, dy, dz = symbolic_calculation_of_gradient(input_args=input_args, target_points=np.zeros((3, 4)))

    np.testing.assert_allclose(dx, np.ones(4))
    np.testing.assert_allclose(dy, np.zeros(4))
    assert dz.shape == (4,)
