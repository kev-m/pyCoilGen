import numpy as np
from scipy.io import loadmat

# Logging
import logging

# Local imports
from sub_functions.constants import *

log = logging.getLogger(__name__)

def load_matlab(filename):
    mat = loadmat(filename+'.mat')  # , struct_as_record=False)
    return mat


def print_structure(mat_input, indent_char=' ', indent=None):
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
        if CURRENT_LEVEL >= DEBUG_VERBOSE:
            log.debug(" No more children")
        return data
    key = parts[0]

    if '[' in key:
        # Handle indexing if the element is an array
        key, key_index = key.split('[')
        key_index = int(key_index.rstrip(']'))
    else:
        key_index = 0

    if CURRENT_LEVEL >= DEBUG_VERBOSE:
        log.debug(" - Searching for %s[%d]", key, key_index)
    dtype = data.dtype
    fields = dtype.fields
    if fields is not None:
        field_index = 0
        for field_name, field_info in fields.items():
            if field_name == key:
                if CURRENT_LEVEL >= DEBUG_VERBOSE:
                    log.debug("Found: %s", field_name)
                if key_len == 1:
                    if CURRENT_LEVEL >= DEBUG_VERBOSE:
                        log.debug("Returning data[%s][%d]", field_name, key_index)
                    return data[field_name][key_index][0]
                else:
                    return _get_element_by_name_internal(data[field_index][key_index][0], parts[1:])
            field_index += 1
        raise AttributeError(f"Key {key} not found!")
    else:
        if CURRENT_LEVEL >= DEBUG_VERBOSE:
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
    if CURRENT_LEVEL >= DEBUG_BASIC:
        log.debug(" Searching for %s", name)
    elements = name.split('.')
    result = _get_element_by_name_internal(data=np_data_array, parts=elements, transpose=transpose)
    if transpose and isinstance(result, (np.ndarray)):
        return result.T
    return result
