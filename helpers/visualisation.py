import numpy as np
from PIL import Image, ImageDraw

# Logging
import logging

log = logging.getLogger(__name__)

def visualize_vertex_connections(vertices, image_x_size, image_path, mesh_uv=None):
    if mesh_uv is not None:
        shape_uv = np.shape(mesh_uv)
        log.debug(" mesh_uv shape: %s", shape_uv)
        if len(shape_uv) == 3:
            log.debug(" mesh_uv shape: Extracting sub-array")
            mesh_uv = mesh_uv[0]
            log.debug(" new mesh_uv shape: %s", np.shape(mesh_uv))
    else:
        log.debug(" mesh_uv shape: None")


    log.debug(" vertices shape: %s", vertices.shape)
    # Find the midpoint of all vertices
    midpoint = np.mean(vertices, axis=0)

    v_width = np.max(vertices[:,0]) - np.min(vertices[:,0]) + 10
    v_height = np.max(vertices[:,1]) - np.min(vertices[:,1]) + 10

    #print("v_width: ", v_width, ", v_height", v_height)
    # Scale the vertices to fit within the image size
    scaled_vertices = vertices - np.min(vertices, axis=0)
    scaled_vertices *= int((image_x_size - 1) / np.max(scaled_vertices))

    # Translate the scaled vertices based on the midpoint
    image_y_size = int(image_x_size*v_height/v_width)
    translated_vertices = scaled_vertices + np.array([image_x_size, image_y_size]) / image_x_size * v_width - midpoint

    # Create a blank image
    image = Image.new('RGB', (image_x_size, image_y_size), color='white')
    draw = ImageDraw.Draw(image)

    # Draw the vertex connections
    radius_start = 5
    radius_end = 7
    if mesh_uv is not None:
        for uv in mesh_uv:
            x1, y1 = translated_vertices[uv[0]]
            x2, y2 = translated_vertices[uv[1]]
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


# Example usage:
mesh_uv = [(0, 1), (1, 2), (2, 0)]  # Example vertex connections
vertices = np.array([(-1, 0), (1, 1), (2, -1)])  # Example vertex coordinates
image_path = 'vertex_connections.png'  # Output image path

visualize_vertex_connections(vertices, 800, image_path, mesh_uv)
