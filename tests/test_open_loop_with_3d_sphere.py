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


def get_loop():
    uv = np.array([[-0.1389, -0.0605, -0.0722, -0.0605, -0.0694, -0.1058, -0.1135, -0.1175, -0.3352, -0.538, -0.6413,
                    -0.6744, -0.9057, -1.0831, -1.1099, -1.2258, -1.24, -1.2444, -1.2863, -1.2474, -1.244, -1.1276,
                    -1.1171, -1.0943, -0.9137, -0.674, -0.6478, -0.3802, -0.3382, -0.0879, -0.0611, -0.0571, -0.0503,
                    -0.0627, -0.1398, -0.1361, -0.4727, -0.5672, -0.9804, -1.0389, -1.4183, -1.7057, -1.7505,
                    -1.7704, -1.9512, -1.9569, -2.0071, -1.9252, -1.9171, -1.7139, -1.6922, -1.6469, -1.3529,
                    -0.9833, -0.9265, -0.7483, -0.4453, -0.1389],
                   [1.8348, 1.8087, 1.6925, 1.6612, 1.5437, 1.4898, 1.3893, 1.2891, 1.2625, 1.181, 1.1143, 1.0805,
                    0.9077, 0.6817, 0.6423, 0.3701, 0.3334, 0.2993, 0.001, -0.3011, -0.3326, -0.6151, -0.6445, -0.6793,
                    -0.9137, -1.0956, -1.1231, -1.2319, -1.2687, -1.4108, -1.4399, -1.5635, -1.6847, -1.7117,
                    -1.7558, -1.8525, -1.9117, -1.9022, -1.7398, -1.7057, -1.4183, -1.0389, -0.986, -0.9403, -0.4802,
                    -0.4327, 0.0577, 0.5392, 0.5859, 1.022, 1.0654, 1.114, 1.4627, 1.7204, 1.7504, 1.7877, 1.8985,
                    1.8348]])
    v = np.array([[-0.4935, -0.4962, -0.4961, -0.4963, -0.4963, -0.4944, -0.4943, -0.4944, -0.483, -0.4497, -0.433,
                   -0.4232, -0.3536, -0.2634, -0.25, -0.1422, -0.1294, -0.1175, 0, 0.1185, 0.1294, 0.24, 0.25, 0.2616,
                   0.3536, 0.4253, 0.433, 0.4763, 0.483, 0.4956, 0.4971, 0.4969, 0.4968, 0.4965, 0.494, 0.4939,
                   0.483, 0.4734, 0.433, 0.4222, 0.3536, 0.264, 0.25, 0.2381, 0.1294, 0.117, 0, -0.117, -0.1294,
                   -0.2381, -0.25, -0.264, -0.3536, -0.4222, -0.433, -0.4517, -0.483, -0.4935],
                  [0.0492, 0.0292, 0.0298, 0.0279, 0.028, 0.0428, 0.0433, 0.0428, 0.1294, 0.2097, 0.25, 0.2628, 0.3536,
                   0.4227, 0.433, 0.4777, 0.483, 0.4845, 0.5, 0.4844, 0.483, 0.4372, 0.433, 0.4241, 0.3536, 0.2601,
                   0.25, 0.1454, 0.1294, 0.0336, 0.0223, 0.0236, 0.0245, 0.0263, 0.0453, 0.0467, 0.1294, 0.1524, 0.25,
                   0.264, 0.3536, 0.4222, 0.433, 0.438, 0.483,
                   0.4846, 0.5, 0.4846, 0.483, 0.438, 0.433, 0.4222, 0.3536, 0.264, 0.25, 0.2049, 0.1294, 0.0492],
                  [-0.6, -0.5662, -0.45, -0.4176, -0.3, -0.2504, -0.15, -0.0496, -0.0499, -0.0501, -0.0261, -0.0185,
                   -0.0193, -0.0194, -0.0159, -0.0159, -0.0143, -0.0138, -0.0138, -0.0127, -0.0131, -0.0124, -0.0138,
                   -0.0168, -0.0157, -0.0146, -0.0206, -0.0199, -0.0385, -0.15, -0.1758, -0.3, -0.4217, -0.45, -0.5025,
                   -0.6, -0.6949, -0.7213, -0.7213, -0.7297, -0.7297, -0.7297, -0.7334, -0.7351, -0.7351, -0.7357,
                   -0.7357, -0.7357, -0.7351, -0.7351, -0.7334, -0.7297, -0.7297, -0.7297, -0.7211, -0.6939, -0.6939,
                   -0.6]])

    loop = Shape3D(uv=uv, v=v)
    return loop

def test_add_nearest_ref_point_to_curve():
    result = np.load('tests/test_data/test_add_nearest_ref_point_to_curve1.npy', allow_pickle=True)[0]
    curve_track_in = result['curve_points_in'] # 3,57
    #curve_track_in = get_loop() # 3,58
    sphere_center_p = result['sphere_center']
    sphere_center = [[sphere_center_p[0]],[sphere_center_p[1]],[sphere_center_p[2]]] # Python to MATLAB

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



def test_open_loop_with_3d_sphere():
    result = np.load('tests/test_data/test_open_loop_with_3d_sphere1.npy', allow_pickle=True)[0]
    loop = result['curve_points']
    sphere_center_p = result['sphere_center']
    sphere_center = [[sphere_center_p[0]],[sphere_center_p[1]],[sphere_center_p[2]]] # Python to MATLAB
    sphere_diameter = result['sphere_diameter']

    # TODO: Remove DEBUG ONLY 'debug=....'
    debug_data = np.load('tests/test_data/test_open_loop_with_3d_sphere1_debug.npy', allow_pickle=True)[0]
    opened_loop, uv_cut, cut_points = open_loop_with_3d_sphere(
        curve_points_in=loop, sphere_center=sphere_center, sphere_diameter=sphere_diameter,
        debug_data=debug_data)

    assert compare(opened_loop.uv, result['opened_loop'].uv)
    assert compare(opened_loop.v, result['opened_loop'].v)
    assert compare(uv_cut, result['uv_cut'])
    assert compare(cut_points.uv, result['cut_points'].uv)
    assert compare(cut_points.v, result['cut_points'].v)

    log.debug("---here --")


def make_data(filename):
    mat_data = load_matlab(filename)
    coil_parts = mat_data['coil_layouts'].out.coil_parts
    open_loop_with_3d_sphere_data = coil_parts.groups[0].loops[0].open_loop_with_3d_sphere
    intput_data = open_loop_with_3d_sphere_data.inputs
    output_data = open_loop_with_3d_sphere_data.outputs

    result = {}
    debug_base = output_data.curve_points_in
    result['curve_points'] = Shape3D(uv=debug_base.curve_points1.uv, v=debug_base.curve_points1.v)
    result['sphere_center'] = debug_base.sphere_center
    result['sphere_diameter'] = debug_base.sphere_diameter
    result['opened_loop'] = Shape3D(uv=output_data.opened_loop.uv, v=output_data.opened_loop.v)
    result['uv_cut'] = output_data.uv_cut
    result['cut_points'] = Shape3D(uv=output_data.cut_points.uv, v=output_data.cut_points.v)
    np.save('tests/test_data/test_open_loop_with_3d_sphere1.npy', [result])

    debug_data = {}
    #debug_data['xxxx'] = debug_base.xxxx
    debug_data['points_to_delete'] = debug_base.points_to_delete
    debug_data['number_points'] = debug_base.number_points
    #debug_data['parts_avg_dist'] = debug_base.parts_avg_dist
    debug_data['inside_sphere_ind_unique1'] = debug_base.inside_sphere_ind_unique1
    debug_data['first_sphere_penetration_locations'] = debug_base.first_sphere_penetration_locations
    debug_data['second_sphere_penetration_locations'] = debug_base.second_sphere_penetration_locations
    debug_data['first_distances'] = debug_base.first_distances
    debug_data['second_distances'] = debug_base.second_distances
    debug_data['sphrere_crossing_vecs'] = debug_base.sphrere_crossing_vecs
    debug_data['cut_points'] = debug_base.cut_points
    debug_data['shift_ind'] = debug_base.shift_ind
    debug_data['inside_sphere_ind_unique2'] = debug_base.inside_sphere_ind_unique2
    debug_data['curve_points1'] = debug_base.curve_points1
    debug_data['curve_points2'] = debug_base.curve_points2
    debug_data['curve_points3'] = debug_base.curve_points3
    debug_data['first_distances'] = debug_base.first_distances
    debug_data['mean_dist_1'] = debug_base.mean_dist_1
    debug_data['mean_dist_2'] = debug_base.mean_dist_2
    np.save('tests/test_data/test_open_loop_with_3d_sphere1_debug.npy', [debug_data])

    # Data to be used by 'test_add_nearest_ref_point_to_curve1'
    result = {}
    result['curve_points_in'] = Shape3D(v=debug_base.curve_points0.v, uv=debug_base.curve_points0.uv)
    result['sphere_center'] = debug_base.sphere_center
    result['sphere_diameter'] = debug_base.sphere_diameter
    result['near_points'] = debug_base.near_points1
    result['curve_track_out'] = debug_base.curve_points1
    np.save('tests/test_data/test_add_nearest_ref_point_to_curve1.npy', [result])

    debug_data = {}
    #debug_data['target_point'] = debug_base.target_point
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
    debug_data['curve_track_out'] = debug_base.add_nearest_ref_point_to_curve.curve_track_out
    np.save('tests/test_data/test_add_nearest_ref_point_to_curve1_debug.npy', [debug_data])
    

if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    make_data('debug/ygradient_coil')
    test_add_nearest_ref_point_to_curve()
    #test_open_loop_with_3d_sphere()
