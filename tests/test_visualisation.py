############################
# TEST DEBUG: Remove when test works
import logging

# Local imports
import sys
from pathlib import Path
# Add the sub_functions directory to the Python module search path
sub_functions_path = Path(__file__).resolve().parent / '..'
sys.path.append(str(sub_functions_path))
#
############################

import numpy as np

# Code under test
import helpers.visualisation as vs

def test_compare_1d():
    val1 = np.array([0.1])
    val2 = np.array([0.1])
    assert vs.compare(val1, val2)

    val2 = np.array([0.11])
    assert vs.compare(val1, val2) == False
    assert vs.compare(val1, val2, 0.01)

def test_compare_2d():
    val1 = np.array([0.1, 0.2])
    val2 = np.array([0.1, 0.2])
    assert vs.compare(val1, val2)

    val2 = np.array([0.11, 0.2])
    assert vs.compare(val1, val2) == False
    assert vs.compare(val1, val2, 0.01)

    val2 = np.array([0.11, 0.21])
    assert vs.compare(val1, val2) == False
    assert vs.compare(val1, val2, 0.01)


def test_compare_contains_1d():
    # Trivial
    val1 = np.array([0.1, 0.2, 0.3])
    val2 = np.array([0.1, 0.2, 0.3])
    assert vs.compare_contains(val1, val2)

    val2 = np.array([0.11, 0.2, 0.3])
    assert vs.compare_contains(val1, val2) == False
    #assert vs.compare_contains(val1, val2, 0.01)

    # Order does not matter when both are 1D
    val1 = np.array([0.1, 0.2, 0.3])
    val2 = np.array([0.3, 0.2, 0.1])
    assert vs.compare_contains(val1, val2)


def test_compare_contains_2d():
    # Trivial
    val1 = np.array([[0.1, 0.2], [0.2, 0.3]])
    val2 = np.array([[0.1, 0.2], [0.2, 0.3]])
    assert vs.compare_contains(val1, val2)

    # Order matters when both are 2D
    val1 = np.array([[0.2, 0.3], [0.1, 0.2]])
    val2 = np.array([[0.2, 0.1], [0.2, 0.3]])
    assert vs.compare_contains(val1, val2) == False
    val2 = np.array([[0.1, 0.2], [0.3, 0.2]])
    assert vs.compare_contains(val1, val2) == False


    # Order reversed
    val1 = np.array([[0.2, 0.3], [0.1, 0.2]])
    val2 = np.array([[0.1, 0.2], [0.2, 0.3]])
    assert vs.compare_contains(val1, val2)

    val2 = np.array([[0.11, 0.21], [0.21, 0.31]])
    assert vs.compare_contains(val1, val2) == False
    assert vs.compare_contains(val1, val2, 0.01)


if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    test_compare_1d()
    test_compare_contains_1d()
    test_compare_contains_2d()



