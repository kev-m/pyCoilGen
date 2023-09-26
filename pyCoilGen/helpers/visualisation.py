from inspect import currentframe
import numpy as np
from PIL import Image, ImageDraw

# Logging
import logging

log = logging.getLogger(__name__)


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno


def passify_matlab(matlab_thing, magic=0):
    """
    Some MATLAB items are arrays when the have more than 1 element, else are scalars.

    This functions ensures that these scalars are converted into arrays.
    """
    if magic == 1:
        if matlab_thing.shape == (2,):
            result = np.asarray([[matlab_thing[0]], [matlab_thing[1]]])
            return result
        else:
            return matlab_thing
    # Magic 2: Turn the matlab_thing array into a 1,n array
    if magic == 2:
        m_array = np.empty((1), dtype=object)
        m_array[0] = matlab_thing
        return m_array
    if isinstance(matlab_thing, np.ndarray):
        return matlab_thing
    return np.asarray([matlab_thing])


def _compare_list(index, instance1, instance2, double_tolerance=1e-10, equal_nan=True):
    """
    Compare two instances for equality with optional double tolerance.

    Args:
        instance1 (list): The first instance to compare.
        instance2 (ndarray): The second instance to compare.
        double_tolerance (float): Tolerance for comparing floating-point values.

    Returns:
        bool: True if the instances are equal, False otherwise.

    Raises:
        TypeError: If the type of `instance1` is not supported.
    """
    if isinstance(instance1, list):
        instance2 = passify_matlab(instance2)
        if len(instance1) != instance2.shape[0]:
            log.error(" Not the same shape at index %d: %s is not %s", index, len(instance1), np.shape(instance2))
            return False
        try:
            if isinstance(instance1[0], list):
                if _compare_list(0, instance1[0], instance2[0]) == False:
                    return False
            else:
                for index in range(len(instance1)):
                    if instance1[index] is None and instance2[index].shape[0] == 0:
                        continue

                    if np.allclose(instance1[index], instance2[index], atol=double_tolerance) == False:
                        log.error(" Not the same value at index [%d]: %s is not %s",
                                  index, instance1[index], instance2[index])
                        return False
        except IndexError:
            return instance2.shape[0] == 0
        return True
    if isinstance(instance1, np.ndarray):
        return compare(instance1, instance2)

    log.error("compare(): Type(%s) is not supported", type(instance1))
    return False


def compare(instance1, instance2, double_tolerance=1e-10, equal_nan=True, fail_result=False):
    """
    Compare two instances for equality with optional double tolerance.

    Args:
        instance1: The first instance to compare.
        instance2: The second instance to compare.
        double_tolerance (float): Tolerance for comparing floating-point values.

    Returns:
        bool: True if the instances are equal, False otherwise.

    Raises:
        TypeError: If the type of `instance1` is not supported.
    """
    if not type(instance1) == type(instance2) and not isinstance(instance1, list):
        log.error(" Not the same type: %s is not %s", type(instance1), type(instance2))
        return fail_result

    if isinstance(instance1, np.ndarray):
        if not instance1.shape == instance2.shape:
            log.error(" Not the same shape: %s is not %s", np.shape(instance1), np.shape(instance2))
            return fail_result
        for index in range(instance1.shape[0]):
            # log.debug(" %d -> %s", index, instance1[index])
            if not np.shape(instance1[index]) == np.shape(instance2[index]):
                log.error(" Not the same shape at index %d: %s is not %s",
                          index, np.shape(instance1[index]), np.shape(instance2[index]))
                return fail_result

            if not np.allclose(instance1[index], instance2[index], atol=double_tolerance, equal_nan=equal_nan):
                if isinstance(instance1[index], np.ndarray):
                    log.error(" Not the same value at index [%d]:\n %s ... is not\n %s ...",
                              index, instance1[index][:5], instance2[index][:5])
                else:
                    log.error(" Not the same value at index [%d]: %s is not %s",
                              index, instance1[index], instance2[index])
                return fail_result
        return True

    if isinstance(instance1, list) and isinstance(instance2, list):
        if len(instance1) != len(instance2):
            log.error(" Not the same shape: %s is not %s", len(instance1), np.shape(instance2))
            return fail_result

        if len(instance1) > 0 and isinstance(instance1[0], list):
            for index in range(len(instance1)):
                if instance1[index] != instance2[index]:
                    log.error(" Not the same value at index [%d]: %s is not %s",
                              index, instance1[index], instance2[index])
                    return fail_result
        return True

    if isinstance(instance1, list) and isinstance(instance2, list):
        if len(instance1) != len(instance2):
            log.error(" Not the same shape: %s is not %s", len(instance1), np.shape(instance2))
            return False

        if len(instance1) > 0 and isinstance(instance1[0], list):
            for index in range(len(instance1)):
                if instance1[index] != instance2[index]:
                    log.error(" Not the same value at index [%d]: %s is not %s",
                              index, instance1[index], instance2[index])
                    return False
        return True

    if isinstance(instance1, list):
        if len(instance1) != instance2.shape[0]:
            log.error(" Not the same shape: %s is not %s", len(instance1), np.shape(instance2))
            return fail_result

        for index in range(len(instance1)):
            if _compare_list(index, instance1[index], instance2[index]) == False:
                log.error(" Not the same value at index [%d]: %s is not %s",
                          index, instance1[index], instance2[index])
                return fail_result
        return True

    if isinstance(instance1, float):
        if np.isclose(instance1, instance2, atol=double_tolerance, equal_nan=equal_nan) == False:
            log.error("Not the same: %f <> %f by %f", instance1, instance2, instance1-instance2)
            return fail_result
        return True

    log.error("compare(): Type(%s) is not supported", type(instance1))
    return False


def compare_contains(array1, array2, double_tolerance=1e-10, strict=True, equal_nan=True):
    """
    Checks if array1 and array2 are the same shape and contain the same elements, row-wise.

    The "strict" parameter determines if element order is important for the array elements.

    Args:
        array1: The first array to compare.
        array2: The second array to compare.
        double_tolerance (float): Tolerance for comparing floating-point values.
        strict (bool): Whether order matters in the sub-elements.

    Returns:
        bool: True if the instances have matching entries, False otherwise.
    """
    # Simplication: Rude implementation if mixed order list and array
    if isinstance(array1, list):
        for item1 in array1:
            found = False
            for item2 in array2:
                if len(item1) == len(item2) and np.allclose(item1, item2, atol=double_tolerance, equal_nan=equal_nan):
                    found = True
                    break
            if found:
                continue
            log.error("Can not find value %s in %s", item1, array2)
            return False
        return True

    if not type(array1) == type(array2):
        log.error(" Not the same type: %s is not %s", type(array1), type(array2))
        return False

    if isinstance(array1, np.ndarray):
        if not array1.shape == array2.shape:
            log.error(" Not the same shape: %s is not %s", np.shape(array1), np.shape(array2))
            return False

        # Handle if array1 / array2 are simple arrays
        # e.g. [0.1, 0.2, 0.3] and [0.3, 0.2, 0.1]
        if not isinstance(array1[0], np.ndarray):
            for value in array1:
                if np.isclose(array2, value, atol=double_tolerance, equal_nan=equal_nan).any() == False:
                    log.error("Can not find value %s in %s", value, array2)
                    return False
            return True

        # Handle if array1 and array2 are arrays of arrays
        # e.g. [[0.2, 0.3], [0.1, 0.2]] and [[0.1, 0.2], [0.2, 0.3]] are "equal"
        # e.g. [[0.21, 0.3], [0.11, 0.2]] and [[0.1, 0.2], [0.2, 0.3]] are "equal" within tolerance.
        for index in range(array1.shape[0]):
            # log.debug(" %d -> %s", index, instance1[index])
            if not np.shape(array1[index]) == np.shape(array2[index]):
                log.error(" Not the same shape at index %d: %s is not %s",
                          index, np.shape(array1[index]), np.shape(array2[index]))
                return False

        for index in range(array1.shape[0]):
            if strict:
                found = False
                item = array1[index]
                for index2 in range(array2.shape[0]):
                    if np.allclose(item, array2[index2], atol=double_tolerance, equal_nan=equal_nan):
                        found = True
                        array2 = np.delete(array2, index2, axis=0)
                        break
                if found == False:
                    log.error(" Can not find value at index [%d] %s:\n %s ... and \n %s ...",
                              index, item, array1[:5], array2[:5])
                    return False
            else:
                # Check that every item in arr1 is in arr2
                found = True
                for subitem in array1[index]:
                    if np.isclose(subitem, array2[index], atol=double_tolerance, equal_nan=equal_nan).any() == False:
                        found = False
                        break
                if found == False:
                    log.error(" Can not find value at index [%d] %s:\n %s ... and \n %s ...",
                              index, array1[index], array1[:5], array2[:5])
                    return False

        return True

    log.error("compare_contains(): Type(%s) is not supported", type(array1))
    return False


def visualize_vertex_connections(vertices2_or_3d, image_x_size, image_path, boundaries=None):
    """ Blah..
    """
    if boundaries is not None:
        shape_edges = np.shape(boundaries)
        # log.debug(" faces shape: %s", shape_edges)
        if len(shape_edges) == 3:
            log.debug(" Edges shape: Extracting sub-array")
            boundaries = boundaries[0]
            log.debug(" new edges shape: %s", np.shape(boundaries))

    if vertices2_or_3d.shape[1] == 3:
        vertices_2d = vertices2_or_3d[:, :2]
    else:
        vertices_2d = vertices2_or_3d
    # Find the midpoint of all vertices
    midpoint = np.mean(vertices_2d, axis=0)

    v_width = np.max(vertices_2d[:, 0]) - np.min(vertices_2d[:, 0]) + 10
    v_height = np.max(vertices_2d[:, 1]) - np.min(vertices_2d[:, 1]) + 10

    # print("v_width: ", v_width, ", v_height", v_height)
    # Scale the vertices to fit within the image size
    scaled_vertices = vertices_2d - np.min(vertices_2d, axis=0)
    scaled_vertices *= int((image_x_size - 1) / np.max(scaled_vertices))

    # Translate the scaled vertices based on the midpoint
    image_y_size = int(image_x_size*v_height/v_width)
    translated_vertices = scaled_vertices + np.array([image_x_size, image_y_size]) / image_x_size * v_width - midpoint

    # Create a blank image
    image = Image.new('RGB', (image_x_size+20, image_y_size+20), color='white')
    draw = ImageDraw.Draw(image)

    # Draw the vertex connections
    radius_start = 2
    radius_end = 3
    if boundaries is not None:
        for boundary in boundaries:
            edges = len(boundary)
            for edge_index in range(edges-1):
                x1, y1 = translated_vertices[boundary[edge_index]]
                x2, y2 = translated_vertices[boundary[edge_index+1]]
                draw.line([(x1, y1), (x2, y2)], fill='black')
                draw.ellipse((x1 - radius_start, y1 - radius_start, x1 + radius_start, y1 + radius_start), fill='red')
                draw.ellipse((x2 - radius_end, y2 - radius_end, x2 + radius_end, y2 + radius_end), fill='blue')
    else:
        for index in range(vertices_2d.shape[0]-1):
            x1, y1 = translated_vertices[index]
            draw.ellipse((x1 - radius_start, y1 - radius_start, x1 + radius_start, y1 + radius_start), fill='red')
            x2, y2 = translated_vertices[index+1]
            draw.ellipse((x2 - radius_end, y2 - radius_end, x2 + radius_end, y2 + radius_end), fill='blue')
            draw.line([(x1, y1), (x2, y2)], fill='black')

    # Save the image
    image.save(image_path)


def visualize_multi_connections(vertices2_or_3d, image_x_size, image_path, connection_list):
    if vertices2_or_3d.shape[1] == 3:
        vertices_2d = vertices2_or_3d[:, :2]
    else:
        vertices_2d = vertices2_or_3d
    log.debug(" vertices shape: %s", vertices_2d.shape)
    # Find the midpoint of all vertices
    midpoint = np.mean(vertices_2d, axis=0)

    v_width = np.max(vertices_2d[:, 0]) - np.min(vertices_2d[:, 0]) + 10
    v_height = np.max(vertices_2d[:, 1]) - np.min(vertices_2d[:, 1]) + 10

    # print("v_width: ", v_width, ", v_height", v_height)
    # Scale the vertices to fit within the image size
    scaled_vertices = vertices_2d - np.min(vertices_2d, axis=0)
    scaled_vertices *= int((image_x_size - 1) / np.max(scaled_vertices))

    # Translate the scaled vertices based on the midpoint
    image_y_size = int(image_x_size*v_height/v_width)
    translated_vertices = scaled_vertices + np.array([image_x_size, image_y_size]) / image_x_size * v_width - midpoint

    # Create a blank image
    image = Image.new('RGB', (image_x_size+20, image_y_size+20), color='white')
    draw = ImageDraw.Draw(image)

    # Draw the vertex connections
    radius_start = 5
    radius_end = 7
    for boundary_lines in connection_list:
        for boundary_line in boundary_lines:
            edges = len(boundary_line)
            for edge_index in range(edges-1):
                x1, y1 = translated_vertices[boundary_line[edge_index]]
                x2, y2 = translated_vertices[boundary_line[edge_index+1]]
                draw.line([(x1, y1), (x2, y2)], fill='black')
                draw.ellipse((x1 - radius_start, y1 - radius_start, x1 + radius_start, y1 + radius_start), fill='red')
                draw.ellipse((x2 - radius_end, y2 - radius_end, x2 + radius_end, y2 + radius_end), fill='blue')

    # Save the image
    image.save(image_path)


def visualize_connections(vertices2d, image_x_size, image_path, connection_list):
    # Find the midpoint of all vertices
    midpoint = np.mean(vertices2d, axis=0)

    v_width = np.max(vertices2d[:, 0]) - np.min(vertices2d[:, 0])
    v_height = np.max(vertices2d[:, 1]) - np.min(vertices2d[:, 1])
    minima = np.min(vertices2d, axis=0)

    # Calculate x-scale
    vertices2d1_scale = (image_x_size / v_width)

    # Translate the scaled vertices based on the midpoint
    image_y_size = int(image_x_size*v_height/v_width)

    # Create a blank image
    image = Image.new('RGB', (image_x_size+20, image_y_size+20), color='white')
    draw = ImageDraw.Draw(image)

    radius_start = 1.6
    radius_stop = 1.4

    for connections in connection_list:
        start_uv = vertices2d[0]
        start_xy = (start_uv - minima) * vertices2d1_scale
        x1 = start_xy[0]
        y1 = start_xy[1]
        draw.ellipse((x1 - radius_start, y1 - radius_start, x1 + radius_start, y1 + radius_start), fill='red')
        for i in range(1, len(connections)):
            stop_uv = vertices2d[i]
            stop_xy = (stop_uv - minima) * vertices2d1_scale
            # log.debug(" %s -> %s", start_xy, stop_xy)
            draw.line([(start_xy[0], start_xy[1]), (stop_xy[0], stop_xy[1])], fill='black')
            start_xy = stop_xy
        x1 = stop_xy[0]
        y1 = stop_xy[1]
        draw.ellipse((x1 - radius_stop, y1 - radius_stop, x1 + radius_stop, y1 + radius_stop), fill='blue')
        # log.debug("-")

    # Save the image
    image.save(image_path)


def visualize_compare_vertices(vertices2d1, vertices2d2, image_x_size, image_path):
    """
    Draw both the vertex arrays onto an image and write the image to file.

    The function draws vectors from vertices2d1[n] to vertices2d2[n] for each vertex.
    The vectors will be visible only as dots if the vertex arrays are identical.

    Args:
        vertices2d1 (ndarray): A 2D array of the vertices projected onto a 2D plance.
        vertices2d2 (ndarray): A 2D array of the vertices projected onto a 2D plance.
        image_x_size (int): The desired width of the output image.
        image_path (string): The desired output path where to write the image.
    """
    # Find the midpoint of all vertices
    midpoint = np.mean(vertices2d1, axis=0)

    v_width = np.max(vertices2d1[:, 0]) - np.min(vertices2d1[:, 0])
    v_height = np.max(vertices2d1[:, 1]) - np.min(vertices2d1[:, 1])
    minima = np.min(vertices2d1, axis=0)

    # Calculate x-scale
    vertices2d1_scale = (image_x_size / v_width)

    # Translate the scaled vertices based on the midpoint
    image_y_size = int(image_x_size*v_height/v_width)

    # Create a blank image
    image = Image.new('RGB', (image_x_size+20, image_y_size+20), color='white')
    draw = ImageDraw.Draw(image)

    radius_start = 1.5
    for i in range(vertices2d1.shape[0]):
        start_uv = vertices2d1[i]
        stop_uv = vertices2d2[i]

        start_xy = (start_uv - minima) * vertices2d1_scale
        stop_xy = (stop_uv - minima) * vertices2d1_scale

        x1 = start_xy[0]
        y1 = start_xy[1]
        draw.line([(x1, y1), (stop_xy[0], stop_xy[1])], fill='black')
        draw.ellipse((x1 - radius_start, y1 - radius_start, x1 + radius_start, y1 + radius_start), fill='red')

    # Save the image
    image.save(image_path)


def visualize_projected_vertices(vertices3d, image_x_size, image_path):
    """
    Project the provided 3D vertex array onto 2D.

    The function draws vectors from vertices3d[n] to vertices3d[n+1] for each vertex.

    Args:
        vertices3d (ndarray): A 3D (m,3) array of the vertices.
        image_x_size (int): The desired width of the output image.
        image_path (string): The desired output path where to write the image.

    Returns:
        vertices2d (ndarray): The projected 2D (m,2) array of vertices.
    """
    # Project the 3D onto 2D using:
    # Project the vertices onto the X-Y plane:  [x,y,z] -> [x+x*z, y+y*z, 0]
    vertices2d = vertices3d[:, :2].copy()  # Copy, otherwise modifies source
    vertices2d[:, 0] += vertices3d[:, 0] * vertices3d[:, 2]
    vertices2d[:, 1] += vertices3d[:, 1] * vertices3d[:, 2]

    # Find the midpoint of all 2D vertices
    midpoint = np.mean(vertices2d, axis=0)

    v_width = np.max(vertices2d[:, 0]) - np.min(vertices2d[:, 0])
    v_height = np.max(vertices2d[:, 1]) - np.min(vertices2d[:, 1])
    minima = np.min(vertices2d, axis=0)

    # Calculate x-scale
    vertices2d1_scale = (image_x_size / v_width)

    # Translate the scaled vertices based on the midpoint
    image_y_size = int(image_x_size*v_height/v_width)

    # Create a blank image
    image = Image.new('RGB', (image_x_size+20, image_y_size+20), color='white')
    draw = ImageDraw.Draw(image)

    radius_start = 1.5
    for i in range(vertices2d.shape[0]-1):
        start_uv = vertices2d[i]
        stop_uv = vertices2d[i+1]

        start_xy = (start_uv - minima) * vertices2d1_scale
        stop_xy = (stop_uv - minima) * vertices2d1_scale

        x1 = start_xy[0]
        y1 = start_xy[1]
        draw.line([(x1, y1), (stop_xy[0], stop_xy[1])], fill='black')
        # draw.ellipse((x1 - radius_start, y1 - radius_start, x1 + radius_start, y1 + radius_start), fill='red')

    # Save the image
    image.save(image_path)

    return vertices2d


def visualize_compare_contours(vertices2d, image_x_size, image_path, contour_list, centres=None):
    """
    Draw the given contour_list onto an image of the specified size and save the image to a file.

    The starting point is drawn with a red circle and the ending point is drawn with a blue circle.

    The vertices2d parameter is only used to compute the image scaling factors.

    Args:
        vertices2d (ndarray): A 2D array of the vertices projected onto a 2D plance.
        image_x_size (int): The desired width of the output image.
        image_path (string): The desired output path where to write the image.
        contour_list (list[ContourLine]): The list of contour lines.
        centres(ndarray): A list of 2D centres to draw in, if provided (nx2)
    """
    # Find the midpoint of all vertices
    midpoint = np.mean(vertices2d, axis=0)

    v_width = np.max(vertices2d[:, 0]) - np.min(vertices2d[:, 0])
    v_height = np.max(vertices2d[:, 1]) - np.min(vertices2d[:, 1])
    minima = np.min(vertices2d, axis=0)

    # Calculate x-scale
    vertices2d1_scale = (image_x_size / v_width)

    # Translate the scaled vertices based on the midpoint
    image_y_size = int(image_x_size*v_height/v_width)

    # Create a blank image
    image = Image.new('RGB', (image_x_size+20, image_y_size+20), color='white')
    draw = ImageDraw.Draw(image)

    radius_start = 1.8
    radius_stop = 1.4

    for contour in contour_list:
        uv = contour.uv.T  # Array of [x,y] pairs
        num_contours = len(uv)
        start_uv = uv[0]
        start_xy = (start_uv - minima) * vertices2d1_scale
        x1 = start_xy[0]
        y1 = start_xy[1]
        draw.ellipse((x1 - radius_start, y1 - radius_start, x1 + radius_start, y1 + radius_start), fill='red')
        for i in range(1, num_contours):
            stop_uv = uv[i]
            stop_xy = (stop_uv - minima) * vertices2d1_scale
            # log.debug(" %s -> %s", start_xy, stop_xy)
            draw.line([(start_xy[0], start_xy[1]), (stop_xy[0], stop_xy[1])], fill='black')
            start_xy = stop_xy
        x1 = stop_xy[0]
        y1 = stop_xy[1]
        draw.ellipse((x1 - radius_stop, y1 - radius_stop, x1 + radius_stop, y1 + radius_stop), fill='blue')
        # log.debug("-")

    if centres is not None:
        radius_stop = 2
        centres_p = centres.T
        for center in centres_p:
            xy = (center - minima) * vertices2d1_scale
            x1 = xy[0]
            y1 = xy[1]
            draw.ellipse((x1 - radius_stop, y1 - radius_stop, x1 + radius_stop, y1 + radius_stop), fill='green')

    # Save the image
    image.save(image_path)


def visualize_faces(tri_vertices2d, image_x_size, image_path, centres=None):
    """
    Draw the given set of faces onto an image of the specified size and save the image to a file.

    The starting point is drawn with a red circle and the ending point is drawn with a blue circle.

    The vertices2d parameter is only used to compute the image scaling factors.

    Args:
        tri_vertices2d (ndarray): A list of 2D arrays of the vertices (nx3x2). Each entry describes one face.
        image_x_size (int): The desired width of the output image.
        image_path (string): The desired output path where to write the image.
        contour_list (list[ContourLine]): The list of contour lines.
        centres(ndarray): A list of 2D centres to draw in, if provided (nx2)
    """
    tri_vertices2d = np.asarray(tri_vertices2d)
    # Find the midpoint of all vertices
    midpoint = np.mean(np.mean(tri_vertices2d, axis=1), axis=0)

    v_width = np.max(tri_vertices2d[:, :, 0]) - np.min(tri_vertices2d[:, :, 0])
    v_height = np.max(tri_vertices2d[:, :, 1]) - np.min(tri_vertices2d[:, :, 1])
    minima = np.min(np.min(tri_vertices2d, axis=1), axis=0)

    # Calculate x-scale
    vertices2d1_scale = (image_x_size / v_width)

    # Translate the scaled vertices based on the midpoint
    image_y_size = int(image_x_size*v_height/v_width)

    # Create a blank image
    image = Image.new('RGB', (image_x_size+20, image_y_size+20), color='white')
    draw = ImageDraw.Draw(image)

    radius_start = 1.8
    radius_stop = 1.4

    for face_corners in tri_vertices2d:
        start_uv = face_corners[0]
        start_xy = (start_uv - minima) * vertices2d1_scale
        for index in range(1, len(face_corners)):
            x1 = start_xy[0]
            y1 = start_xy[1]

            stop_uv = face_corners[index]
            stop_xy = (stop_uv - minima) * vertices2d1_scale
            # log.debug(" %s -> %s", start_xy, stop_xy)
            draw.line([(start_xy[0], start_xy[1]), (stop_xy[0], stop_xy[1])], fill='black')
            start_xy = stop_xy

        start_xy = (face_corners[0] - minima) * vertices2d1_scale
        draw.line([(stop_xy[0], stop_xy[1]), (start_xy[0], start_xy[1])], fill='black')

    if centres is not None:
        radius_stop = 2
        centres_p = centres.T
        for center in centres_p:
            xy = (center - minima) * vertices2d1_scale
            x1 = xy[0]
            y1 = xy[1]
            draw.ellipse((x1 - radius_stop, y1 - radius_stop, x1 + radius_stop, y1 + radius_stop), fill='green')

    # Save the image
    image.save(image_path)


def project_vertex_onto_plane(vertex, pov):
    # Extract the coordinates of the vertex and POV
    x1, y1, z1 = vertex
    _, _, z2 = pov

    # Calculate the projection factor based on the distance between vertex and POV
    projection_factor = z2 / (z2 - z1)

    # Perform the perspective projection onto the X-Y plane
    projected_vertex = np.array([x1, y1, z1]) * projection_factor

    return projected_vertex[:2]  # Return only the X and Y coordinates


def visualize_3D_boundary(boundary_loops, vertices, image_x_size, image_path):

    log.debug(" vertices shape: %s", vertices.shape)
    # Find the midpoint of all vertices
    midpoint = np.mean(vertices, axis=0)

    v_width = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    v_height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    z_depth = np.max(vertices[:, 2]) - np.min(vertices[:, 2])

    log.debug(" - v_width: %s, v_height: %s", v_width, v_height)
    pov = [0, 0, np.max(vertices[:, 2]) + z_depth/10.]

    # scale_x =

    # For each boundary loop:
    for boundary in boundary_loops:
        # For each point in the loop:
        for vertex_index in boundary:
            # get the vertex
            vertex = vertices[vertex_index]
            # project the vertex onto an x-y plane
            log.debug(" -- Vertex: %s, %s", vertex, project_vertex_onto_plane(vertex, pov))


if __name__ == "__main__":
    # Example usage:
    mesh_uv = [(0, 1), (1, 2), (2, 0)]  # Example vertex connections
    vertices = np.array([(-1, 0), (1, 1), (2, -1)])  # Example vertex coordinates
    image_path = 'vertex_connections.png'  # Output image path

    visualize_vertex_connections(vertices, 800, image_path, mesh_uv)
