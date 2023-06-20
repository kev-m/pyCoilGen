# Sketchpad file to compare data from Matlab save file with data computed by current CoilGen implementation.

import json
import numpy as np
from scipy.io import loadmat
# Logging
import logging

DEBUG_VERBOSE = 1
def get_level():
    return 2

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
            
def _get_element_by_name_internal(data, parts, transpose=True):
    key_len = len(parts)
    if key_len == 0:
        if get_level() >= DEBUG_VERBOSE:
            log.debug(" No more children")
        return data
    key = parts[0]

    if '[' in key:
        # Handle indexing if the element is an array
        key, key_index = key.split('[')
        key_index = int(key_index.rstrip(']'))
    else:
        key_index = 0

    if get_level() >= DEBUG_VERBOSE:
        log.debug(" - Searching for %s[%d]", key, key_index)
    dtype = data.dtype
    fields = dtype.fields
    if fields is not None:
        field_index = 0
        for field_name, field_info in fields.items():
            if field_name == key:
                if get_level() >= DEBUG_VERBOSE:
                    log.debug("Found: %s", field_name)
                if key_len == 1:
                    part_data = data[field_name][key_index]
                    log.debug(" Part_data[%s][%d]: %s", field_index, key_index, part_data.dtype)

                    if get_level() >= DEBUG_VERBOSE:
                        log.debug("Returning data[%s][%d]", field_name, key_index)
                    # TODO: Implement Transpose here
                    return data[field_name][key_index][0]
                else:
                    
                    log.debug("---- data.type: %s", data.dtype)
                    if len(data) == len(fields):
                        part_data = data[field_index][key_index]
                        return _get_element_by_name_internal(part_data[0], parts[1:])
                    else:
                        # DEBUG: Problem here
                        if key == 'target_field':
                            log.debug("target_field found!")
                            log.debug(" len: %d", len(data))
                            log.debug(" shape: %s", data.shape)
                            log.debug(" type: %s", data.dtype)
                        return _get_element_by_name_internal(data[key][key_index], parts[1:])
            field_index += 1
        raise AttributeError(f"Key {key} not found!")
    else: # for coil_mesh[0], coil_mesh.uv : dtype('O')
        if get_level() >= DEBUG_VERBOSE:
            log.debug(" --- %s", np.shape(data[key_index]))
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
    result = _get_element_by_name_internal(data=np_data_array, parts=elements, transpose=transpose)
    if transpose and isinstance(result, (np.ndarray)):
        return result.T
    return result

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        log.debug(" ENCODE: %s of type %s", obj, type(obj))
        if isinstance(obj, np.ndarray):
            #return json.JSONEncoder.default(self, obj[0].tolist())   
            return f'{obj.tolist()}'
        return obj.decode('UTF-8')
        #return json.JSONEncoder.default(self, obj)

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
    name = 'out.coil_parts[0].coil_mesh.boundary'
    # name = 'out.coil_parts[0].coil_mesh.faces'
    # name = 'out.coil_parts[0].coil_mesh.faces.faces[0]'
    # name = 'out.coil_parts[0].coil_mesh.unique_vert_inds'
    # name = 'out.coil_parts[0].coil_mesh.uv'
    # name = 'out.coil_parts[0].one_ring_list'
    name = 'out.target_field.b'

    # Get the corresponding numpy element
    target_field = get_element_by_name(x_channel, name, transpose=True)
    log.debug(" -- target_field.dtype : %s", target_field.dtype)
    log.debug("  -- shape: %s", target_field.shape)
    #log.debug("  -- value: %s", target_field)
    #log.debug("  -- value: %s", target_field['coil_mesh_file'])
    #log.debug("  -- value: %s", target_field['cylinder_mesh_parameter_list'])

    #b = target_field#['b']
    #log.debug(" -- b: %s", b.dtype.fields)
    #log.debug(" -- b: %s", b)

    # 2. Extract input parameters structure
    name = 'out.input_data'
    input_data = get_element_by_name(x_channel, name)
    # input_dict = object_to_dict(input_data)
    log.debug("  -- input: type: %s", input_data.dtype)
    log.debug("  -- input: fields: %s", input_data.dtype.fields)
    input_dict = {}
    for field_name, field_info in input_data.dtype.fields.items():
        log.debug(" -- Field %s type: %s", field_name, input_data[field_name].dtype)

        if field_name == 'fieldtype_to_evaluate':
            log.debug(' -- item of interest')

        value = input_data[field_name][0][0]
        if len(value) > 0:
            value = value[0]
            log.debug(" --- Field %s shape: %s", field_name, np.shape(value))
            log.debug(" --- Field %s = %s", field_name, value)
            if len(value) > 1:
                p_value = value.tolist()
            else:
                p_value = value.tolist()[0]

            if isinstance(p_value, bytes):
                p_value = p_value.decode('UTF-8')
                log.debug(" Decoding %s to string: %s",field_name, p_value)
            
            input_dict[field_name] = p_value

    json_obj = json.dumps(input_dict, cls=MyEncoder)
    print(json_obj)


    # 3. Call CoilGen code with equivalent input parameters
