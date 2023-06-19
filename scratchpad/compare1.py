# Sketchpad file to compare data from Matlab save file with data computed by current CoilGen implementation.

import numpy as np
from scipy.io import loadmat
# Logging
import logging


def load_matlab(filename):
    mat = loadmat(filename+'.mat')  # , struct_as_record=False)
    return mat


def print_structure(mat_input, indent_char = ' ', indent = None):
    """
    Prints the structure of all nested elements within a numpy ndarray.

    Args:
        data (numpy.ndarray): The input.
        indent (str): Optional. The current indentation level for printing.
    """
    if indent is None:
        indent = indent_char
    for item in mat_input:
        try:
            dtype = item.dtype
            fields = dtype.fields
            if fields is not None:
                index = 0
                for field_name, field_info in fields.items():
                    print(f'{indent}: {field_name}')
                    try:
                        next_item = item[index]
                        if isinstance(next_item, object):
                            print_structure(next_item, indent+indent_char, indent_char)
                            index += 1
                    except IndexError:
                        return
        except AttributeError:
            return
            
def get_element_by_name_internal(data, parts, transpose=True):
    key_len = len(parts)
    if key_len == 0:
        log.debug(" No more children")
        return data
    key = parts[0]

    if '[' in key:
        # Handle indexing if the element is an array
        key, key_index = key.split('[')
        key_index = int(key_index.rstrip(']'))
    else:
        key_index = 0


    log.debug(" - Searching for %s[%d]", key, key_index)
    dtype = data.dtype
    fields = dtype.fields
    if fields is not None:
        field_index = 0
        for field_name, field_info in fields.items():
            if field_name == key:
                log.debug("Found: %s", field_name)
                if key_len == 1:
                    log.debug("Returning data[%s][%d]", field_name, key_index)
                    return data[field_name][key_index][0]
                else:
                    return get_element_by_name_internal(data[field_index][key_index][0], parts[1:])
            field_index += 1
        raise AttributeError(f"Key {key} not found!")
    else:
        log.debug(" -- %s", np.shape(data[key_index]))
        if transpose and isinstance(data, (np.ndarray)):
            return data.T[key_index]
        return data[key_index]

def get_element_by_name(np_data_array, name, transpose=True):
    """
    Finds and returns the field with the given name from the given np_data_array.

    The field name is specified in dot notation: 'first.second.third'.
    Array indexing is supported: 'first.second[1].third'.

    Args:
        np_data_array (numpy.ndarray): The input.
        name (str): The desired field name in dot notation.
        transpose (bool): Transpose the result, if it is an array.

    Raises:
        AttributeError: If the key is not found.
        IndexError: If the array index is invalid.
    """
    log.debug(" Searching for %s", name)
    elements = name.split('.')
    result = get_element_by_name_internal(data=np_data_array, parts=elements, transpose=transpose)
    if transpose and isinstance(result, (np.ndarray)):
        return result.T
    return result

if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # 1. Load Matlab
    mat_contents = load_matlab('debug/result_y_gradient')
    log.debug("mat_contents: %s", mat_contents.keys())
    x_channel = mat_contents['coil_layouts']

    print_structure(x_channel, '-')

    # Dot-notation name
    name = 'out'
    name = 'out.coil_parts[0]'
    name = 'out.coil_parts[0].coil_mesh'
    # name = 'out.coil_parts[0].coil_mesh.boundary'
    name = 'out.coil_parts[0].coil_mesh.faces'
    # name = 'out.coil_parts[0].coil_mesh.faces.faces[0]'
    # name = 'out.coil_parts[0].coil_mesh.unique_vert_inds'
    # name = 'out.coil_parts[0].coil_mesh.uv'
    # name = 'out.coil_parts[0].one_ring_list'
    name = 'out.input_data'

    # Get the corresponding numpy element
    input_data = get_element_by_name(x_channel, name, transpose=True)
    
    log.debug(" -- input_data.dtype : %s", input_data.dtype)
    log.debug("  -- shape: %s", input_data.shape)
    log.debug("  -- value: %s", input_data)
    #log.debug("  -- value: %s", input_data['coil_mesh_file'])
    #log.debug("  -- value: %s", input_data['cylinder_mesh_parameter_list'])

    # 2. Extract input parameters structure
    # input = matlab_data.input

    # 3. Call CoilGen code with equivalent input parameters
