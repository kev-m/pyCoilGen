from numpy import sum, ndarray, zeros
from os import path

# Logging
import logging


from pyCoilGen.sub_functions.constants import get_level, DEBUG_NONE

log = logging.getLogger(__name__)


def nearest_approaches(point: ndarray, starts: ndarray, ends: ndarray):
    """
    Calculate the nearest approach of a point to arrays of line segments.

    NOTE: Uses MATLAB shape conventions

    Args:
        point (ndarray): The point of interest (3D coordinates) (1,3).
        starts (ndarray): The line segment starting positions (m,3)
        ends (ndarray): The line segment ending positions (m,3)

    Returns:
       distances, diffs (ndarray, ndarray): The nearest approach distances and the diffs array for re-use.
    """
    diffs = ends - starts
    vec_targets2 = point - starts
    t1 = sum(vec_targets2 * diffs, axis=0) / sum(diffs * diffs, axis=0)
    return t1, diffs


def blkdiag(arr1: ndarray, arr2: ndarray) -> ndarray:
    """
    Compute the block diagonal matrix created by aligning the input matrices along the diagonal.

    Args:
        arr1 (ndarray): The first input matrix.
        arr2 (ndarray): The second input matrix.

    Returns:
        ndarray: The block diagonal matrix created by aligning arr1 and arr2 along the diagonal.
    """
    # Get the dimensions of the arrays
    rows1, cols1 = arr1.shape
    rows2, cols2 = arr2.shape

    # Create a larger zero-filled array with the final desired size
    result = zeros((rows1 + rows2, cols1 + cols2))

    # Place the smaller array within the larger one
    result[:rows1, :cols1] = arr1
    result[rows1:, cols1:] = arr2

    return result


# A list of possible paths to try: 'data' in both the local and site-packages installed directories.
__directory_list = [
    path.join('data', 'pyCoilGenData'),
]


def __add_pyCoilGenData(to_list: list):
    try:
        from pyCoilGenData import data_directory
        data_directory_str = data_directory()
        to_list.append(data_directory_str)
        log.debug("Adding '%s' to data search path", data_directory_str)
    except ImportError:
        log.debug("Package 'pyCoilGenData' is not installed. Unable to retrieve the data directory. Install with 'pip install pycoilgen_data'")


def find_file(file_directory: str, file_name: str) -> str:
    """
    Iterates through candidate paths to find a file on the file system.

    Args:
        file_directory (str): The default directory to search in.
        file_name (str): The filename to load.

    Returns:
        new_file_name (str): The actual file name, if it has been found.

    Raises:
        FileNotFoundError: If the file can not be found anywhere.
    """
    path_list = __directory_list.copy()
    __add_pyCoilGenData(path_list)
    dir_path = path.join(file_directory, file_name)
    if path.exists(dir_path):
        log.debug("Found '%s'", dir_path)
        return dir_path
    for new_path in path_list:
        new_file_name = path.join(new_path, dir_path)
        if path.exists(new_file_name):
            log.debug("Found '%s'", new_file_name)
            return new_file_name
    raise FileNotFoundError(f"Unable to find {dir_path} in local path or {__directory_list}")


def title_to_filename(title_str: str):
    """Convert a title string into a valid filename string."""
    result = title_str
    for char in "\n.:\\/ ":
        result = result.replace(char, '_')
    return result
