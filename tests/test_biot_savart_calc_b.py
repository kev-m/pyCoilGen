import numpy as np

from sub_functions.data_structures import TargetField
# Code under test
from sub_functions.process_raw_loops import biot_savart_calc_b

def test_biot_savart_calc_b_trivial():
    wire_path = np.asarray(([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0])).reshape(3,2)
    coords = np.asarray(([0.0], [1.0], [0.0])).reshape(3,1)
    target_field = TargetField(coords=coords)

    ###################################################################################
    # Function under test
    result = biot_savart_calc_b(wire_path, target_field)
    ###################################################################################
    expected = np.asarray(([0.000000e+00, 0.000000e+00, 1.000000e-07]))
    assert np.all(expected == result.T)

def test_biot_savart_calc_b_arrays():
    # Less than 1000 elements, wire_path is processed in one go
    elements = 501 # 500 segments
    span = np.arange(0.0, 1.0, 1.0/elements, dtype=np.float64)
    span /= np.max(span)    # 0 -> 1.0
    span -= 0.5             # -0.5 -> 0.5
    zero_arr = np.zeros_like(span)
    wire_path = np.asarray((span, zero_arr, zero_arr)).reshape(3,-1)
    coords = np.asarray(([0.0], [1.0], [0.0])).reshape(3,-1)
    target_field = TargetField(coords=coords)

    ###################################################################################
    # Function under test
    result = biot_savart_calc_b(wire_path, target_field)
    ###################################################################################
    expected = np.asarray(([0.000000e+00, 0.000000e+00, 8.9442747608e-08]))
    assert np.allclose(expected, result.T)

    # More than 1000 elements, wire_path is split into portions
    elements = 1501 # 1500 segments
    span = np.arange(0.0, 1.0, 1.0/elements, dtype=np.float64)
    span /= np.max(span)    # 0 -> 1.0
    span -= 0.5             # -0.5 -> 0.5
    zero_arr = np.zeros_like(span)
    wire_path = np.asarray((span, zero_arr, zero_arr)).reshape(3,-1)
    coords = np.asarray(([0.0], [1.0], [0.0])).reshape(3,-1)
    target_field = TargetField(coords=coords)

    ###################################################################################
    # Function under test
    result = biot_savart_calc_b(wire_path, target_field)
    ###################################################################################
    assert np.allclose(expected, result.T)


def test_biot_savart_calc_b_arrays2():
    # Split target field into 100 elements
    elements = 100 # 1500 segments
    span = np.arange(0.0, 1.0, 1.0/elements, dtype=np.float64)
    span /= np.max(span)    # 0 -> 1.0
    result_span = span.copy()
    span -= 0.5             # -0.5 -> 0.5
    zero_arr = np.zeros_like(span)
    wire_path = np.asarray((span, zero_arr, zero_arr)).reshape(3,-1)
    coords = np.asarray((zero_arr, span, zero_arr)).reshape(3,-1)
    target_field = TargetField(coords=coords)

    ###################################################################################
    # Function under test
    result = biot_savart_calc_b(wire_path, target_field)
    ###################################################################################

    assert np.allclose(-2.8284631974e-07, result[2,0])          # First
    assert np.allclose(4.8554163453e-05, result[2,50])          # Middle
    assert np.allclose(2.9160871578e-07, result[2,elements-1])  # Last




if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)


# Input: 
#   wire_path: -0.5, 0.5
#   field:  0.0, 1.0, 0.0
#   current: 1000.0
# Result: 0.000000e+00, 0.000000e+00, 1.000000e-04