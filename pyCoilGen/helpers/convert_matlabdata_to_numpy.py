# https://stackoverflow.com/questions/874461/read-mat-files-in-python
# Note: The CoilGen matlab files have the following header:
# MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Fri Oct 28 15:47:12 2022

import scipy.io
import numpy as np

# Logging
import logging


def load_matlab(filename):
    mat = scipy.io.loadmat(filename+'.mat')
    return mat


def save_numpy(filename, data):
    result = np.save(filename+'.npy', data)
    return result


if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # mat = load_matlab('../CoilGen/target_fields/intraoral_dental_target_field')
    # log.debug(" Loaded: %s", mat)
    # result = save_numpy('target_fields/intraoral_dental_target_field', mat)

    mat = load_matlab('debug/result_y_gradient')
    log.debug(" Loaded: %s", mat)
    result = save_numpy('debug/result_y_gradient', mat)
