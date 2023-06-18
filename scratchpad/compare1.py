# Sketchpad file to compare data from Matlab save file with data computed by current CoilGen implementation.

import numpy as np
from scipy.io import loadmat
# Logging
import logging


def load_matlab(filename):
    mat = loadmat(filename+'.mat')  # , struct_as_record=False)
    return mat


def explore(mat_contents, what):
    log.debug(f" -- {what} Type: %s", type(mat_contents[what]))
    # log.debug( f" -- {what} Content: %s", mat_contents[what])


def print_nested_structure(data, indent=''):
    """
    Prints the structure of all nested elements within a numpy.ndarray.

    Args:
        data (numpy.ndarray): The input array.
        indent (str): Optional. The current indentation level for printing.
    """
    if isinstance(data, np.ndarray):
        print(f'{indent}Array(shape={data.shape}, dtype={data.dtype})')
        for i, item in enumerate(data):
            if isinstance(item, (np.number)):
                break
            print(f'{indent}Index1 {i}:')
            print_nested_structure(item, indent + '  ')
    elif isinstance(data, (list, tuple)):
        print(f'{indent}List or Tuple (length={len(data)})')
        for i, item in enumerate(data):
            print(f'{indent}Index2 {i}:')
            print_nested_structure(item, indent + '  ')
    elif isinstance(data, dict):
        print(f'{indent}Dictionary (keys={list(data.keys())})')
        for key, value in data.items():
            print(f'{indent}Key: {key}')
            print_nested_structure(value, indent + '  ')
    elif isinstance(data, (np.uint16)):
        pass
    else:
        print(f'{indent}Value: {data.dtype}')
        # print(f'{indent}Value: not shown')
        print_nested_structure(data[0], indent + '  ')


def find_field(mat_input, seek_field_name, indent = ' '):
    for item in mat_input:
        try:
            dtype = item.dtype
            fields = dtype.fields
            if fields is not None:
                index = 0
                for field_name, field_info in fields.items():
                    print(f'{indent}Field: {field_name}')
                    try:
                        next_item = item[index]
                        if isinstance(next_item, object):
                            find_field(next_item, seek_field_name, indent+' ')
                            index += 1
                    except IndexError:
                        return
        except AttributeError:
            return
            
def get_element_by_name_internal(data, parts):
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
                    # index 13 is out of bounds for axis 0 with size 1
                    #return data[key_index][field_index]
                else:
                    return get_element_by_name_internal(data[field_index][key_index][0], parts[1:])
            field_index += 1
        raise AttributeError(f"Key {key} not found!")
    else:
        log.error("Not handled")
        raise Exception("Not implemented yet")
    log.debug(" Dunno")
    return  get_element_by_name_internal(current_element, parts[1:])

def get_element_by_name(data, name):
    log.debug(" Searching for %s", name)
    elements = name.split('.')
    return get_element_by_name_internal(data=data, parts=elements)

if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # 1. Load Matlab
    mat_contents = load_matlab('debug/result_x_gradient')
    log.debug("mat_contents: %s", mat_contents.keys())
    # explore(mat_contents, '__globals__')
    explore(mat_contents, 'x_channel')

    x_channel = mat_contents['x_channel']
    log.debug(f" -- x_channel[0] Type: %s", type(x_channel))
    # log.debug( f" -- x_channel[0] : %s", x_channel[0][0])
    log.debug(f" -- x_channel.dtype : %s", x_channel[0].dtype)

    # print_nested_structure(x_channel)
    find_field(x_channel, 'input')

    # xxx = str(x_channel[0])
    # print(dir(xxx))
    yyy = x_channel[0][0][0][0][0][0][0][0][0][0][0]
    # print(dir(yyy))
    xxx = str(yyy.dtype)
    # print (xxx[0:100])
    print("xxx:", xxx)
    print("yyy['boundary'].dtype:", yyy['boundary'].dtype)
    print(x_channel['out'][0].dtype)

    # Dot-notation name
    name = 'out'
    name = 'out.coil_parts[0]'
    name = 'out.coil_parts[0].coil_mesh'
    name = 'out.coil_parts[0].coil_mesh.boundary'
    #name = 'out.coil_parts[0].coil_mesh.faces'

    # Get the corresponding numpy element
    input_data = get_element_by_name(x_channel, name)
    
    log.debug(" -- input_data.dtype : %s", input_data.dtype)
    log.debug("  -- value: %s", input_data)

    # 2. Extract input parameters structure
    # input = matlab_data.input

    # 3. Call CoilGen code with equivalent input parameters
