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
from sub_functions.remove_points_from_loop import remove_points_from_loop

def test_remove_points_from_loop():
    result = np.load('tests/test_data/test_remove_points_from_loop1.npy', allow_pickle=True)[0]

    for index1, level in enumerate(result['levels']):
        for index2, connection in enumerate(level['connections']):
            #log.debug(" Level: %d, connection: %d", index1, index2)
            inputs = connection['inputs'] 
            outputs = connection['outputs']

            loop = Shape3D(inputs.loop.uv, inputs.loop.v)
            points_to_remove = inputs.points_to_remove
            boundary_threshold = inputs.boundary_threshold

            # Function under test
            loop_out_uv, loop_out_v = remove_points_from_loop(loop, points_to_remove, boundary_threshold)

            assert compare(np.array(loop_out_uv), outputs.loop_out_uv)
            assert compare(np.array(loop_out_v), outputs.loop_out_v)
            #log.debug(" Result: %s", compare(np.array(loop_out_uv), outputs.loop_out_uv))


def make_data(filename):
    mat_data = load_matlab(filename)
    coil_parts = mat_data['coil_layouts'].out.coil_parts
    top_debug = coil_parts.interconnect_among_groups.level_debug

    # Test data for remove_points_from_loop
    result = {'levels' : []}
    for index1, level_debug in enumerate(top_debug.connections):
        level_entry = {'connections' : []}
        for index2, remove_debug in enumerate(level_debug.remove_points_debug):
            connection_entry = {}
            connection_entry['inputs'] = remove_debug.inputs
            connection_entry['outputs'] = remove_debug.outputs
            level_entry['connections'].append(connection_entry)
        result['levels'].append(level_entry)
    np.save('tests/test_data/test_remove_points_from_loop1.npy', [result])

if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    # make_data('debug/ygradient_coil')
    test_remove_points_from_loop()
