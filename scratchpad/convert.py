# System imports
import sys
from pathlib import Path
import numpy as np
import json

# Logging
import logging

# Add the sub_functions directory to the Python module search path
# Only required for the development environment
import sys
from pathlib import Path
sub_functions_path = Path(__file__).resolve().parent / '..'
sys.path.append(str(sub_functions_path))

## Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE
from pyCoilGen.sub_functions.data_structures import DataStructure, TargetField, CoilSolution
from pyCoilGen.helpers.extraction import load_matlab, print_structure


log = logging.getLogger(__name__)


def load_numpy(filename) -> CoilSolution:
    [solution] = np.load(filename, allow_pickle=True)
    return solution


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
    np.save(f'data/pyCoilGenData/Pre_Optimized_Solutions/{data_name}.npy', [data])


def preoptimzed_breast_coil():
    data_name = 'source_data_breast_coil'
    convert_matlab_to_python(data_name)
    blah = np.load(f'data/pyCoilGenData/Pre_Optimized_Solutions/{data_name}.npy', allow_pickle=True)
    pass


def preoptimzed_SVD_coil():
    data_name = 'source_data_SVD_coil'
    convert_matlab_to_python(data_name)
    blah = np.load(f'data/pyCoilGenData/Pre_Optimized_Solutions/{data_name}.npy', allow_pickle=True)
    pass


def dump_biot_savart_data_p():
    """
    Write out the wirepath and target_field data to text file so that they can be loaded by C algorithm to verify
    the Python implementation of the Biot-Savart algorithm.
    """
    which = 'ygradient_coil_0_5'
    solution = load_numpy(f'debug/{which}_20.npy')
    coil_part = solution.coil_parts[0]

    # Equation needs: wire current, wire segments, target field co-ordinates
    # current = wire_path.currents and coil_part.contour_step
    # segments = wire_path.seg_coords
    # target_field = target_field.coords

    wire_elements = coil_part.wire_path.v.T                         # 3,1539 -> 1539,3
    coords = wire_elements                                           # 1539,3
    seg_coords = (wire_elements[:-1, :] + wire_elements[1:, :]) / 2   # 1538,3    The middle of the segment
    currents = wire_elements[1:, :] - wire_elements[:-1, :]           # 1538,3    The vector of the segment
    contour_step = coil_part.contour_step                           # float

    solution = load_numpy(f'debug/{which}_21.npy')
    coil_part = solution.coil_parts[0]

    target_field = solution.target_field.coords.T                   # 257,3
    target_result = coil_part.field_by_layout.T                      # 257,3

    """
    Compatible with: https://pypi.org/project/biot-savart/
    Coil Format
    The coil is represented as a series of (X,Y,Z) coordinates which define the vertices of the coil spatially, along
    with an additional coordinate (I) which defines the amount of current flowing from that vertex to the next one.

    Plaintext file
    The file contains each (X,Y,Z) entry on a new line
    The format of each line should be "x,y,z\n"    
    """
    with open(f'../magnetic_field_calculator/data/{which}_wire_p.txt', 'w') as my_file:
        my_file.write(f'{len(coords)}\n')
        for index, coord in enumerate(coords):
            my_file.write(f'{coord[0]},{coord[1]},{coord[2]}\n')

    with open(f'../magnetic_field_calculator/data/{which}_target_p.txt', 'w') as my_file:
        my_file.write(f'{len(target_field)}\n')
        for index, coord in enumerate(target_field):
            my_file.write(f'{coord[0]},{coord[1]},{coord[2]}\n')

    with open(f'../magnetic_field_calculator/data/{which}_result_p.txt', 'w') as my_file:
        my_file.write(f'{coil_part.contour_step}\n')
        my_file.write(f'{len(target_result)}\n')
        for index, vector in enumerate(target_result):
            my_file.write(f'{vector[0]},{vector[1]},{vector[2]}\n')


def dump_biot_savart_data_m():
    """
    Write out the wirepath and target_field data to text file so that they can be loaded by C algorithm to verify
    the Python implementation of the Biot-Savart algorithm.
    """
    which = 'ygradient_coil_0_5'
    matlab_data = load_matlab(f'debug/{which}')
    m_out = matlab_data['coil_layouts'].out

    m_c_part = m_out.coil_parts
    m_debug = m_c_part.evaluate_field_errors

    # Equation needs: wire current, wire segments, target field co-ordinates
    # current = wire_path.currents and coil_part.contour_step
    # segments = wire_path.seg_coords
    # target_field = target_field.coords

    wire_elements = m_debug.debug1.input.wire_path.v.T              # 3,1539 -> 1539,3
    coords = wire_elements                                          # 1539,3
    seg_coords = (wire_elements[:-1, :] + wire_elements[1:, :]) / 2 # 1538,3    The middle of the segment
    currents = wire_elements[1:, :] - wire_elements[:-1, :]         # 1538,3    The vector of the segment
    contour_step = m_c_part.contour_step                            # float

    target_field = m_debug.debug2.target_field.coords.T             # 257,3

    target_result = m_c_part.field_by_layout.T                      # 257,3

    """
    Compatible with: https://pypi.org/project/biot-savart/
    Coil Format
    The coil is represented as a series of (X,Y,Z) coordinates which define the vertices of the coil spatially, along
    with an additional coordinate (I) which defines the amount of current flowing from that vertex to the next one.

    Plaintext file
    The file contains each (X,Y,Z) entry on a new line
    The format of each line should be "x,y,z\n"    
    """
    with open(f'../magnetic_field_calculator/data/{which}_wire_m.txt', 'w') as my_file:
        my_file.write(f'{len(coords)}\n')
        for index, coord in enumerate(coords):
            my_file.write(f'{coord[0]},{coord[1]},{coord[2]}\n')

    with open(f'../magnetic_field_calculator/data/{which}_target_m.txt', 'w') as my_file:
        my_file.write(f'{len(target_field)}\n')
        for index, coord in enumerate(target_field):
            my_file.write(f'{coord[0]},{coord[1]},{coord[2]}\n')

    with open(f'../magnetic_field_calculator/data/{which}_result_m.txt', 'w') as my_file:
        my_file.write(f'{m_c_part.contour_step}\n')
        my_file.write(f'{len(target_result)}\n')
        for index, vector in enumerate(target_result):
            my_file.write(f'{vector[0]},{vector[1]},{vector[2]}\n')

def convert_target_field():
    data_name = 'intraoral_dental_target_field'
    mat_data = load_matlab(f'../CoilGen/target_fields/{data_name}')
    data = mat_data['straight_grid_out']
    coords = data.coords # 3,30248
    lin_x = data.lin_x # 30248
    lin_y = data.lin_y # 30248
    lin_z = data.lin_z # 30248
    phi = data.phi
    r = data.r
    z_pot = data.z_pot

    b_field = {'coords': coords, 'lin_x' : lin_x, 'lin_y': lin_y, 'lin_z': lin_z, 'phi': phi, 'r' : r, 'z_pot' : z_pot}
    np.save(f'data/pyCoilGenData/target_fields/{data_name}.npy', [b_field])

def save_bfield_file(filename: str, coords: np.ndarray, vector_field: np.ndarray):
    data = {
      'coords': coords,       # Assuming that coords is a (3,m) array of float.
      'b_field': vector_field # Assuming that vector_field is either an (m,) or (3,m) array of float.
      } 
    np.save(f'target_fields/{filename}.npy', [data], allow_pickle=True)



if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # preoptimzed_breast_coil()
    # preoptimzed_SVD_coil()
    # dump_biot_savart_data_p()
    # dump_biot_savart_data_m()
    # convert_target_field()

    coords = np.ones((3,10), dtype=float)
    values = np.ones((10), dtype=float)
    save_bfield_file('ones', coords=coords, vector_field=values)