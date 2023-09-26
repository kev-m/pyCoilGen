import numpy as np
from pyCoilGen.helpers.pyshull import PySHull

# Logging
import logging

log = logging.getLogger(__name__)


class Triangulate():
    """
    A wrapper that provides Delaunay functionality, hiding the implementation.
    """

    def __init__(self, vertices: np.ndarray) -> None:
        """
        Create an instance of Delaunay triangulation from the provided vertices.

        """
        self._vertices = vertices.copy()
        self._triangles = PySHull(vertices)

    def get_triangles(self):
        return self._triangles

    def get_vertices(self):
        return self._vertices


if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    points = [[0.0, 0.006427876096865392, 0.00984807753012208, 0.008660254037844387, 0.0034202014332566887, -0.0034202014332566865, -0.008660254037844388, -0.009848077530122082, -0.006427876096865396, -2.4492935982947064e-18],
              [0.01, 0.007660444431189781, 0.0017364817766693042, -0.0049999999999999975, -0.009396926207859084, -0.009396926207859084, -0.004999999999999997, 0.0017364817766692998, 0.007660444431189778, 0.01]]

    vertices = np.array(points).T
    d = Triangulate(vertices)
    log.debug(" edges: %s", d.get_triangles())
    for edge in d.get_triangles():
        log.debug(" edges: %s", edge)
