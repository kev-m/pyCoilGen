# System imports
import sys
from pathlib import Path
import numpy as np
import json

# Logging
import logging

# Local imports
# Add the sub_functions directory to the Python module search path
sub_functions_path = Path(__file__).resolve().parent / '..'
print(sub_functions_path)
sys.path.append(str(sub_functions_path))

# Local imports
from sub_functions.constants import *
from helpers.extraction import load_matlab, print_structure
from sub_functions.data_structures import DataStructure, TargetField

log = logging.getLogger(__name__)


def convert_matlab_to_python(data_name):
    """
    Convert source_data_breast_coil.mat to source_data_breast_coil.npy

    All data is converted and transposed from MATLAB format to Python format.
    """
    mat_data = load_matlab(f'../CoilGen/Pre_Optimized_Solutions/{data_name}')

    # CoilMesh
    mesh_data = mat_data['coil_mesh']
    faces = mesh_data.faces.T - 1
    vertices = mesh_data.vertices.T
    mesh = DataStructure(faces=faces, vertices=vertices)

    # stream_function
    stream_function = np.asarray(mat_data['stream_function'], dtype=float)

    # target_field
    target_field_data = mat_data['target_field']
    target_field = TargetField(b=target_field_data.b.T, coords=target_field_data.coords.T)

    data = DataStructure(coil_mesh=mesh, stream_function=stream_function, target_field=target_field)
    np.save(f'Pre_Optimized_Solutions/{data_name}.npy', [data])


def preoptimzed_breast_coil():
    data_name = 'source_data_breast_coil'
    convert_matlab_to_python(data_name)
    blah = np.load(f'Pre_Optimized_Solutions/{data_name}.npy', allow_pickle=True)
    pass

def preoptimzed_SVD_coil():
    data_name = 'source_data_SVD_coil'
    convert_matlab_to_python(data_name)
    blah = np.load(f'Pre_Optimized_Solutions/{data_name}.npy', allow_pickle=True)
    pass

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    preoptimzed_breast_coil()
    preoptimzed_SVD_coil()