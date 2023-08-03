import json
import numpy as np

# Hack code
# Set up paths: Add the project root directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Test support
from helpers.extraction import load_matlab
from sub_functions.data_structures import Shape3D, DataStructure
from helpers.visualisation import compare


# Code under test
from sub_functions.open_loop_with_3d_sphere import open_loop_with_3d_sphere, add_nearest_ref_point_to_curve


def test_open_loop_with_3d_sphere():
    result = np.load('tests/test_data/test_open_loop_with_3d_sphere1.npy', allow_pickle=True)[0]
    loop = result['curve_points_in']
    sphere_center = result['sphere_center']
    sphere_diameter = result['sphere_diameter']

    opened_loop, uv_cut, cut_points = open_loop_with_3d_sphere(
        curve_points_in=loop, sphere_center=sphere_center, sphere_diameter=sphere_diameter
    )

    assert compare(cut_points.uv, result['cut_points'].uv)
    assert compare(cut_points.v, result['cut_points'].v)
    assert compare(opened_loop.uv, result['opened_loop'].uv)
    assert compare(opened_loop.v, result['opened_loop'].v)
    assert compare(uv_cut, result['uv_cut'])


def test_add_nearest_ref_point_to_curve():
    result = np.load('tests/test_data/test_add_nearest_ref_point_to_curve1.npy', allow_pickle=True)[0]
    curve_track_in = result['curve_points_in']  # 3,57
    sphere_center = result['sphere_center']

    # Function under test
    curve_track_out, near_points = add_nearest_ref_point_to_curve(curve_track_in, sphere_center)

    near_points_data = result['near_points']
    assert compare(near_points.uv, near_points_data.uv)  # Pass
    assert compare(near_points.v, near_points_data.v)   # Pass

    curve_track_data = result['curve_track_out']  # 3x58
    assert compare(curve_track_out.uv, curve_track_data.uv)  # Pass
    assert compare(curve_track_out.v, curve_track_data.v)   # Pass


def make_data(filename):
    mat_data = load_matlab(filename)
    coil_parts = mat_data['coil_layouts'].out.coil_parts
    open_loop_with_3d_sphere_data = coil_parts.groups[0].loops[0].open_loop_with_3d_sphere
    input_data = open_loop_with_3d_sphere_data.inputs
    output_data = open_loop_with_3d_sphere_data.outputs

    debug_base = output_data.curve_points_out

    # Test data for test_open_loop_with_3d_sphere
    result = {}
    result['curve_points_in'] = Shape3D(v=debug_base.curve_points0.v, uv=debug_base.curve_points0.uv)
    centre = input_data.sphere_center
    result['sphere_center'] = np.array([[centre[0]], [centre[1]], [centre[2]]])
    result['sphere_diameter'] = input_data.sphere_diameter
    result['opened_loop'] = Shape3D(uv=output_data.opened_loop.uv, v=output_data.opened_loop.v)
    result['uv_cut'] = output_data.uv_cut
    result['cut_points'] = Shape3D(uv=output_data.cut_points.uv, v=output_data.cut_points.v)
    np.save('tests/test_data/test_open_loop_with_3d_sphere1.npy', [result])

    # Data to be used by 'test_add_nearest_ref_point_to_curve1'
    result = {}
    result['curve_points_in'] = Shape3D(v=debug_base.curve_points0a.v, uv=debug_base.curve_points0a.uv)
    centre = input_data.sphere_center
    result['sphere_center'] = np.array([[centre[0]], [centre[1]], [centre[2]]])
    result['sphere_diameter'] = debug_base.sphere_diameter
    result['near_points'] = debug_base.near_points1
    result['curve_track_out'] = debug_base.curve_points1  # 3x58
    np.save('tests/test_data/test_add_nearest_ref_point_to_curve1.npy', [result])


if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    make_data('debug/ygradient_coil')
    test_add_nearest_ref_point_to_curve()
    test_open_loop_with_3d_sphere()
