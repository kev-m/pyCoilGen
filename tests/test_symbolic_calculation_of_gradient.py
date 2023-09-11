import numpy as np

from sub_functions.data_structures import DataStructure, TargetField
# Code under test
from sub_functions.define_target_field import symbolic_calculation_of_gradient


def test_symbolic_calculation_of_gradient():
    input_args = DataStructure(debug=1, field_shape_function='x + y**2')
    target_field = np.full((3, 10), 3)
    result = symbolic_calculation_of_gradient(input_args=input_args, target_field=target_field)

    # TODO: test the result

    # Define the target field shape
    target_points = target_field
    def field_func(x, y, z): return eval(input_args.field_shape_function)
    target_field3 = np.zeros_like(target_points)
    target_field3[2, :] = field_func(target_points[0, :], target_points[1, :], target_points[2, :])

    # TODO: test the result
