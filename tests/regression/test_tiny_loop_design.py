"""A complete design with a loop that is smaller than the interconnection cut."""
import tempfile

import numpy as np
import pytest

from harness_cases import CASES
from harness_run import run_case


@pytest.mark.regression
def test_design_with_a_loop_inside_the_cut_sphere_completes():
    """On this mesh a 7 mm loop used to stop the design with 'no overlap between cut sphere and loop'."""
    CASES['tiny_loop'] = {**CASES['biplanar_x_tikhonov'], 'levels': 16,
                          'biplanar_mesh_parameter_list': [0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0.2]}
    try:
        with tempfile.TemporaryDirectory() as work_dir:
            _, quality = run_case('tiny_loop', work_dir)
    finally:
        del CASES['tiny_loop']
    assert np.isfinite(quality['gradient_max_dev'])
