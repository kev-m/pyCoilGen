import numpy as np
from PIL import Image, ImageDraw

# Logging
import logging

log = logging.getLogger(__name__)


def compare(instance1, instance2, double_tolerance = 0.001):
    if not type(instance1) == type(instance2):
        log.debug(" Not the same type: %s is not %s", type(instance1), type(instance2))
        return False

    if isinstance(instance1, np.ndarray):
        if not instance1.shape == instance2.shape:
            log.debug(" Not the same shape: %s is not %s", np.shape(instance1), np.shape(instance2))
            return False
        log.debug(" Shape[0] %s", instance1.shape[0])
        for index in range(instance1.shape[0]):
            log.debug(" %d -> %s", index, instance1[index])
            if not np.allclose(instance1[index], instance2[index], atol=double_tolerance):
                log.debug(" Not the same value at index [%d]: %s is not %s", index, instance1[index], instance2[index])
                return False
            return True

    log.debug(" %s is not supported", instance1.dtype)
    return False


def visualize_vertex_connections(vertices3d, image_x_size, image_path, edges=None):
    if edges is not None:
        shape_edges = np.shape(edges)
        log.debug(" faces shape: %s", shape_edges)
        if len(shape_edges) == 3:
            log.debug(" Edges shape: Extracting sub-array")
            edges = edges[0]
            log.debug(" new edges shape: %s", np.shape(edges))
    else:
        log.debug(" Edges shape: None")

    vertices = vertices3d[:, :2]
    log.debug(" vertices shape: %s", vertices.shape)
    # Find the midpoint of all vertices
    midpoint = np.mean(vertices, axis=0)

    v_width = np.max(vertices[:, 0]) - np.min(vertices[:, 0]) + 10
    v_height = np.max(vertices[:, 1]) - np.min(vertices[:, 1]) + 10

    # print("v_width: ", v_width, ", v_height", v_height)
    # Scale the vertices to fit within the image size
    scaled_vertices = vertices - np.min(vertices, axis=0)
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
    if edges is not None:
        for edge in edges:
            x1, y1 = translated_vertices[edge[0]]
            x2, y2 = translated_vertices[edge[1]]
            draw.line([(x1, y1), (x2, y2)], fill='black')
            draw.ellipse((x1 - radius_start, y1 - radius_start, x1 + radius_start, y1 + radius_start), fill='red')
            draw.ellipse((x2 - radius_end, y2 - radius_end, x2 + radius_end, y2 + radius_end), fill='blue')
    else:
        for index in range(vertices.shape[0]-1):
            x1, y1 = translated_vertices[index]
            draw.ellipse((x1 - radius_start, y1 - radius_start, x1 + radius_start, y1 + radius_start), fill='red')
            x2, y2 = translated_vertices[index+1]
            draw.ellipse((x2 - radius_end, y2 - radius_end, x2 + radius_end, y2 + radius_end), fill='blue')
            draw.line([(x1, y1), (x2, y2)], fill='black')

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
