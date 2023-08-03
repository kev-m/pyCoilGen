import json
import numpy as np

# Hack code
# Set up paths: Add the project root directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Test support
from helpers.extraction import get_element_by_name, load_matlab, get_and_show_debug
from sub_functions.data_structures import Shape3D, DataStructure
from helpers.visualisation import compare

# Code under test
from sub_functions.open_loop_with_3d_sphere import open_loop_with_3d_sphere, add_nearest_ref_point_to_curve


def test_open_loop_with_3d_sphere():
    result = np.load('tests/test_data/test_open_loop_with_3d_sphere1.npy', allow_pickle=True)[0]
    loop = result['curve_points_in']
    sphere_center = result['sphere_center']
    sphere_diameter = result['sphere_diameter']

    # TODO: Remove DEBUG ONLY 'debug=....'
    debug_data = np.load('tests/test_data/test_open_loop_with_3d_sphere1_debug.npy', allow_pickle=True)[0]
    opened_loop, uv_cut, cut_points = open_loop_with_3d_sphere(
        curve_points_in=loop, sphere_center=sphere_center, sphere_diameter=sphere_diameter,
        debug_data=debug_data)

    assert compare(cut_points.uv, result['cut_points'].uv)
    assert compare(cut_points.v, result['cut_points'].v)
    assert compare(opened_loop.uv, result['opened_loop'].uv)
    assert compare(opened_loop.v, result['opened_loop'].v)
    assert compare(uv_cut, result['uv_cut'])


def test_add_nearest_ref_point_to_curve():
    result = np.load('tests/test_data/test_add_nearest_ref_point_to_curve1.npy', allow_pickle=True)[0]
    curve_track_in = result['curve_points_in'] # 3,57
    sphere_center = result['sphere_center']

    # Function under test
    debug_data = np.load('tests/test_data/test_add_nearest_ref_point_to_curve1_debug.npy', allow_pickle=True)[0]
    curve_track_out, near_points = add_nearest_ref_point_to_curve(curve_track_in, sphere_center, debug_data)

    result = np.load('tests/test_data/test_add_nearest_ref_point_to_curve1.npy', allow_pickle=True)[0]
    near_points_data = result['near_points']
    assert compare(near_points.uv, near_points_data.uv) # Pass
    assert compare(near_points.v, near_points_data.v)   # Pass

    curve_track_data = result['curve_track_out'] # 3x58
    assert compare(curve_track_out.uv, curve_track_data.uv) # Pass
    assert compare(curve_track_out.v, curve_track_data.v)   # Pass

def make_data(filename):
    mat_data = load_matlab(filename)
    coil_parts = mat_data['coil_layouts'].out.coil_parts
    open_loop_with_3d_sphere_data = coil_parts.groups[0].loops[0].open_loop_with_3d_sphere
    input_data = open_loop_with_3d_sphere_data.inputs
    output_data = open_loop_with_3d_sphere_data.outputs

    debug_base = output_data.curve_points_out

    log.debug("1 inputs = inputs? %s", compare(input_data.curve_points_in.v, coil_parts.groups[0].loops[0].v))
    log.debug("2 inputs = inputs? %s", compare(input_data.curve_points_in.v, debug_base.curve_points0.v))

    # Test data for test_open_loop_with_3d_sphere
    result = {}
    result['curve_points_in'] = Shape3D(v=debug_base.curve_points0.v, uv=debug_base.curve_points0.uv)
    centre = input_data.sphere_center
    result['sphere_center'] = np.array([[centre[0]],[centre[1]],[centre[2]]])
    result['sphere_diameter'] = input_data.sphere_diameter
    result['opened_loop'] = Shape3D(uv=output_data.opened_loop.uv, v=output_data.opened_loop.v)
    result['uv_cut'] = output_data.uv_cut
    result['cut_points'] = Shape3D(uv=output_data.cut_points.uv, v=output_data.cut_points.v)
    np.save('tests/test_data/test_open_loop_with_3d_sphere1.npy', [result])

    # Debug data for test_open_loop_with_3d_sphere
    debug_data = {}
    #debug_data['xxxx'] = debug_base.xxxx
    debug_data['curve_points_in'] = Shape3D(v=debug_base.curve_points0.v, uv=debug_base.curve_points0.uv)
    debug_data['curve_points_in2'] = Shape3D(v=input_data.curve_points_in.v, uv=input_data.curve_points_in.uv)
    debug_data['sphere_center'] = np.array([[centre[0]],[centre[1]],[centre[2]]])
    debug_data['sphere_diameter'] = input_data.sphere_diameter
    debug_data['points_to_delete'] = debug_base.points_to_delete
    debug_data['curve_points0a'] = debug_base.curve_points0a
    debug_data['near_points1'] = debug_base.near_points1
    debug_data['cut_points_out'] = Shape3D(uv=output_data.cut_points.uv, v=output_data.cut_points.v)
    debug_data['inside_sphere_ind1'] = debug_base.inside_sphere_ind1.astype(bool)
    debug_data['inside_sphere_ind2'] = debug_base.inside_sphere_ind2.astype(bool)
    debug_data['curve_points1b'] = debug_base.curve_points1b
    debug_data['inside_sphere_ind3'] = debug_base.inside_sphere_ind3.astype(bool)
    debug_data['number_points'] = debug_base.curve_points0a.number_points
    #debug_data['parts_avg_dist'] = debug_base.parts_avg_dist
    debug_data['inside_sphere_ind_unique1'] = debug_base.inside_sphere_ind_unique1.astype(int)
    debug_data['first_sphere_penetration_locations'] = debug_base.first_sphere_penetration_locations
    debug_data['second_sphere_penetration_locations'] = debug_base.second_sphere_penetration_locations
    debug_data['first_distances'] = debug_base.first_distances
    debug_data['second_distances'] = debug_base.second_distances
    debug_data['sphere_crossing_vecs'] = debug_base.sphrere_crossing_vecs
    debug_data['cut_points'] = debug_base.cut_points
    debug_data['shift_ind'] = debug_base.shift_ind
    debug_data['inside_sphere_ind_unique2'] = debug_base.inside_sphere_ind_unique2
    debug_data['curve_points1'] = debug_base.curve_points1
    debug_data['curve_points2'] = debug_base.curve_points2
    debug_data['curve_points3'] = debug_base.curve_points3
    debug_data['first_distances'] = debug_base.first_distances
    debug_data['mean_dist_1'] = debug_base.mean_dist_1
    debug_data['mean_dist_2'] = debug_base.mean_dist_2
    debug_data['uv_cut_array'] = debug_base.uv_cut_array
    np.save('tests/test_data/test_open_loop_with_3d_sphere1_debug.npy', [debug_data])

    # Data to be used by 'test_add_nearest_ref_point_to_curve1'
    result = {}
    result['curve_points_in'] = Shape3D(v=debug_base.curve_points0a.v, uv=debug_base.curve_points0a.uv)
    centre = input_data.sphere_center
    result['sphere_center'] = np.array([[centre[0]],[centre[1]],[centre[2]]])
    result['sphere_diameter'] = debug_base.sphere_diameter
    result['near_points'] = debug_base.near_points1
    result['curve_track_out'] = debug_base.curve_points1 # 3x58
    np.save('tests/test_data/test_add_nearest_ref_point_to_curve1.npy', [result])

    debug_data = {}
    #debug_data['target_point'] = debug_base.target_point
    debug_data['curve_points_in'] = Shape3D(v=input_data.curve_points_in.v, uv=input_data.curve_points_in.uv)
    debug_data['curve_points'] = debug_base.curve_points1
    debug_data['near_points'] = debug_base.near_points1
    #debug_data['xxxx'] = debug_base.add_nearest_ref_point_to_curve.xxxx
    debug_data['curve_track'] = debug_base.add_nearest_ref_point_to_curve.curve_track
    debug_data['t1'] = debug_base.add_nearest_ref_point_to_curve.t1
    debug_data['t2'] = debug_base.add_nearest_ref_point_to_curve.t2
    debug_data['t3'] = debug_base.add_nearest_ref_point_to_curve.t3
    debug_data['seg_starts'] = debug_base.add_nearest_ref_point_to_curve.seg_starts
    debug_data['seg_ends'] = debug_base.add_nearest_ref_point_to_curve.seg_ends
    debug_data['all_dists'] = debug_base.add_nearest_ref_point_to_curve.all_dists
    debug_data['min_ind_seq'] = debug_base.add_nearest_ref_point_to_curve.min_ind_seq
    debug_data['all_near_points'] = debug_base.add_nearest_ref_point_to_curve.all_near_points
    debug_data['near_points'] = debug_base.add_nearest_ref_point_to_curve.near_points
    debug_data['curve_track_out1'] = debug_base.add_nearest_ref_point_to_curve.curve_track_out1
    debug_data['curve_track_out2'] = debug_base.add_nearest_ref_point_to_curve.curve_track_out2
    np.save('tests/test_data/test_add_nearest_ref_point_to_curve1_debug.npy', [debug_data])
    

if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    make_data('debug/ygradient_coil')
    test_add_nearest_ref_point_to_curve()
    test_open_loop_with_3d_sphere()
