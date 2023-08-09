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
from sub_functions.process_raw_loops import process_raw_loops


if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    #make_data('debug/ygradient_coil')
    #test_add_nearest_ref_point_to_curve()
    #test_open_loop_with_3d_sphere()
    #brute_test_process_raw_loops_brute()
