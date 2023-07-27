# System imports
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

# Mesh implementation
import trimesh

# Logging
import logging

# Local imports
from sub_functions.constants import *

log = logging.getLogger(__name__)


# Generic solution to print data classes
def as_string(class_instance):
    properties = vars(class_instance)
    properties_str = ""
    for prop, value in properties.items():
        properties_str += f"{prop}: {value}\n"
    return properties_str


# Generic class with named attributes
class DataStructure:
    """
    Used to create a generic data structure with named attributes.

    Args:
        kwargs (values): name=value pairs of attributes, e.g. a=1, b=2, etc.

    Returns:
        DataStructure (object): an object with attributes, xxx.a, xxx.b, etc.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Mesh class that encapsulates and hides a specific implementation.
class Mesh:
    """
    Represents a mesh without exposing the underlying mesh implementation.

    Args:
        vertices (ndarray, optional): A float array of vertices [x, y, z].
        faces (ndarray, optional): An int array of face indices into the vertices array.
        trimesh_obj (Trimesh, optional): An instance of Trimesh representing the mesh.

    Raises:
        ValueError: If neither vertices and faces, nor trimesh_obj are provided.

    Returns:
        Mesh (object): An abstract mesh.
    """

    def __init__(self, trimesh_obj=None, vertices=None, faces=None):
        if trimesh_obj is not None:
            self.trimesh_obj = trimesh_obj
        elif vertices is not None and faces is not None:
            self.trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            raise ValueError("Either vertices and faces, or trimesh_obj must be provided.")

        # Known properties
        # Assigned in read_mesh
        self.normal_rep = None  # Representative normal for the mesh ([x,y,z])
        # Calculated in parameterize_mesh
        self.v = None           # (n,3) : The array of mesh vertices (n, [x,y,z]).
        self.fn = None          # (n,3) : The face normals (n, [x,y.z]).
        self.n = None           # (n,3) : The vertex normals (n, [x,y.z]).
        self.uv = None          # Vertices, UV texture matrix (n, [x,y,z=0])
        self.boundary = None    # List of 1D lists of vertex indices along mesh boundaries (m,[i])

    def recreate(self, vertices, faces):
        self.trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.cleanup()

    def cleanup(self):
        """
        Clean up the mesh by removing duplicate and unused vertices.
        """
        # Perform simplification operations
        self.trimesh_obj.remove_duplicate_faces()
        self.trimesh_obj.remove_unreferenced_vertices()

    def __str__(self):
        return as_string(self)

    @staticmethod
    def load_from_file(filename):
        """
        Load a mesh from a file.

        Args:
            filename (str): The path to the mesh file.

        Returns:
            Mesh: An instance of Mesh representing the loaded mesh.
        """
        trimesh_obj = trimesh.load_mesh(filename)
        return Mesh(trimesh_obj=trimesh_obj)

    def get_vertices(self):
        """
        Returns the vertices of the mesh.

        Returns:
            ndarray: The float array of vertices [x, y, z].
        """
        return self.trimesh_obj.vertices.view(np.ndarray)

    def get_faces(self):
        """
        Returns the faces of the mesh.

        Returns:
            ndarray: The int array of face indices into the vertices array.
        """
        return self.trimesh_obj.faces.view(np.ndarray)

    def face_normals(self):
        """
        Get the normals of each face in the mesh.

        Returns:
            ndarray: An array of face normals with shape (num_faces, 3).
        """
        return self.trimesh_obj.face_normals

    def vertex_normals(self):
        """
        Get the normals of each vertex in the mesh.

        Returns:
            ndarray: An array of vertex normals with shape (num_faces, 3).
        """
        return self.trimesh_obj.vertex_normals

    def display(self):
        """
        Display the mesh
        """
        return self.trimesh_obj.show()

    def save_to_file(self, filename):
        """
        Save this mesh to a file.
        """
        raise Exception("Not implemented!")

    def refine(self, inplace=False):
        """
        Increase the resolution of the mesh by splitting face.

        Args:
            inplace (bool): Specify to modify the existing Mesh. If false, returns a new Mesh.


        Returns:
            Mesh: The refined Mesh.
        """
        mesh = self.trimesh_obj.subdivide()
        if inplace:
            self.trimesh_obj = mesh
            return self

        return Mesh(mesh)

    def boundary_edges(self):
        """
        Get the boundary face indices of the mesh.

        Returns:
            ndarray: An array of boundary face indices.
        """
        boundary = self.trimesh_obj.facets_boundary
        # DEBUG
        log.debug(" - boundary_edges: shape: %s", np.shape(boundary))
        # if len(np.shape(boundary)) == 3:
        #    log.debug(" boundary: Extracting sub-array")
        #    boundary = boundary[0]

        return boundary

    def vertex_faces(self):
        """
        Get all the vertex face connections.

        For each vertex, fetch the indices of the faces that it is connected to.

        Returns:
            ndarray: An array of arrays of vertex indices.
        """
        num_vertices = self.v.shape[0]
        node_triangles = np.empty(num_vertices, dtype=object)  # []
        # NOTE: Not calculated the same as MATLAB
        # Extract the indices of all the triangles connected to each vertex.
        node_triangles_tri = self.trimesh_obj.vertex_faces
        # This creates an n,m array where m is padded with -1's. See
        # https://trimsh.org/trimesh.base.html#trimesh.base.Trimesh.vertex_faces
        # Iterate and create reduced numpy arrays
        node_triangles = np.array([row[row != -1] for row in node_triangles_tri], dtype=object)
        return node_triangles

# Generated for calculate_basis_functions


class BasisElement:
    stream_function_potential: float
    triangles: List[int]                # node_triangles x ?
    one_ring: np.ndarray                # node_triangles x 1
    area: np.ndarray                    # node_triangles x 1
    face_normal: np.ndarray             # node_triangles x 3
    triangle_points_ABC: np.ndarray     # node_triangles x 3
    current: np.ndarray                 # node_triangles x 3


class UnarrangedLoop:
    """
    Represents an unarranged loop in the coil mesh.

    Used by calc_contours_by_triangular_potential_cuts
    """

    def __init__(self):
        self.edge_inds = []
        self.uv = None
        self.current_orientation: float = None

    def add_edge(self, edge):
        self.edge_inds.append(edge)

    def add_uv(self, uv):
        if self.uv is None:
            self.uv = np.zeros((1, 2), dtype=float)
            self.uv[0] = uv
            return
        self.uv = np.vstack((self.uv, [uv]))


class UnarrangedLoopContainer:
    def __init__(self):
        self.loop: List[UnarrangedLoop] = []


@dataclass
class UnsortedPoints:
    """
    Represents unsorted contours in the coil mesh.

    Used by calc_contours_by_triangular_potential_cuts
    """
    potential: float = None
    edge_ind = None
    uv = None


@dataclass
class RawPart:
    """
    Represents the unprocessed collection of points.

    Used by calc_contours_by_triangular_potential_cuts
    """
    unsorted_points: List[UnsortedPoints] = None
    unarranged_loops: List[UnarrangedLoop] = None


@dataclass
class ContourLine:
    """
    Represents a contour line

    Used by calc_contours_by_triangular_potential_cuts
    """
    v: np.ndarray = None   # 3D co-ordinates of the contour (process_raw_loops) (3,m)
    uv: np.ndarray = None  # 2D co-ordinates of the contour (process_raw_loops) (2,2)
    potential: float = None  # Potential value of the contour
    current_orientation: int = None


@dataclass
class CoilPart:
    coil_mesh: Mesh = None
    basis_elements: List[BasisElement] = None
    resistance_matrix: np.ndarray = None    # num_vertices x num_vertices
    raw: RawPart = None
    contour_lines: List[ContourLine] = None
    field_by_loops: np.ndarray = None       # Placeholder (evaluate_loop_significance in process_raw_loops)
    combined_loop_field: np.ndarray = None  # Placeholder (evaluate_loop_significance in process_raw_loops) (3,m)
    loop_significance: np.ndarray = None    # Per contour line (evaluate_loop_significance in process_raw_loops) (n)
    combined_loop_length: float = 0.0       # Length of contour lines (process_raw_loops)
    pcb_track_width: float = 0.0            # PCB track width (find_minimal_contour_distance)
    loop_groups: List[int] = None           # Topological groups (topological_loop_grouping)
    group_levels: np.ndarray = None         # ??? (topological_loop_grouping)
    level_positions: np.ndarray = None      # ??? (topological_loop_grouping)
    groups: List[ContourLine] = None        # Topological groups (topological_loop_grouping)


class CoilSolution:
    """
    Represents a high-level CoilGen solution.

    Attributes:
        coil_parts (list): A list of mesh parts that make up the coil surface.
        target_field: The target field associated with the CoilSolution.
    """

    def __init__(self):
        self.coil_parts: List[CoilPart] = []
        self.target_field = None
        self.optimisation = OptimisationParameters()

    def __str__(self):
        return as_string(self)


# Used by define_target_field
@dataclass
class TargetField:
    """
    To be defined.
    """
    b: np.ndarray = None       # (3,num vertices)
    coords: np.ndarray = None  # (3,num vertices)
    weights = None              # (num vertices)
    target_field_group_inds = None  # (num vertices)
    target_gradient_dbdxyz = None  # (3,num vertices)

    def __str__(self):
        return as_string(self)

# Used in calculate_gradient


@dataclass
class LayoutGradient:
    dBxdxyz: np.ndarray
    dBydxyz: np.ndarray
    dBzdxyz: np.ndarray
    gradient_in_target_direction: np.ndarray = None
    mean_gradient_in_target_direction: float = None
    std_gradient_in_target_direction: float = None

# Used by process_raw_loops


@dataclass
class WirePart:
    """
    To be defined.
    """
    coord: List[np.ndarray] = None
    seg_coords: List[np.ndarray] = None
    currents: List[np.ndarray] = None

    def __str__(self):
        return as_string(self)


# Used in topological_loop_grouping
@dataclass
class Shape2D:  # Used in topological_loop_grouping
    uv: np.ndarray = None  # 2D co-ordinates of the shape (2,n)


@dataclass
class Shape3D:  # Used in topological_loop_grouping
    uv: np.ndarray = None  # 2D co-ordinates of the shape (2,n)
    v: np.ndarray = None  # 3D co-ordinates of the shape (3,n)


@dataclass
class TopoGroup:
    loops: List[ContourLine] = None  # Assigned in topological_loop_grouping
    cutshape: List[Shape2D] = None
    opened_loop: List[Shape3D] = None


#
#
#  Generated data classes that are not (yet) used.
#
#

@dataclass
class OptimisationParameters:
    preoptimization_hash = None
    optimized_hash = None
    use_preoptimization_temp = False
    use_optimized_temp = False


@dataclass
class UnsortedPoint:
    """
    Represents an unsorted point in the coil mesh.
    """
    potential: float


@dataclass
class Loop:
    """
    Represents a loop in the coil mesh.
    """
    uv: np.ndarray
    edge_inds: np.ndarray
    current_orientation: int


@dataclass
class ParameterizedMesh:
    """
    Represents the parameterized mesh.
    """
    f: np.ndarray
    uv: np.ndarray


@dataclass
class GradientData:
    """
    Unknown
    TODO: find usage.
    """
    mean_gradient_strength: float
    gradient_out: np.ndarray


@dataclass
class LocalOpeningGab:
    """
    Unknown, check against LoopCalculationInput
    TODO: find usage.
    """
    point_1: int
    point_2: int
    opening_gab: float


@dataclass
class CalcRotationMatrixResult:
    """
    Represents the result of the calculation of a rotation matrix.
    """
    rot_mat_out: np.ndarray


@dataclass
class CalcLocalOpeningGabResult:
    """
    Represents the result of the calculation of local opening gab.
    """
    local_opening_gab: float


@dataclass
class CalcLocalOpeningGabOutput:
    """
    Unknown, possible duplicate of above.
    TODO: find usage.
    """
    local_opening_gab: float


@dataclass
class LoopCalculationInput:
    """
    Represents the input data for loop calculation.
    """
    loop: Loop
    point_1: int
    point_2: int
    opening_gab: float


@dataclass
class CalcLocalOpeningGab2Input:
    """
    Unknown, might be duplicate of LoopCalculationInput.
    Possibly use PotentialSortedCutPoints instead and decompose.
    TODO: find usage.
    """
    loop: Loop
    cut_point: np.ndarray
    cut_direction: Tuple[float, float, float]
    opening_gab: float


@dataclass
class PotentialSortedCutPoints:
    """
    Unknown. Poosibly use this instead of CalcLocalOpeningGab2Input
    TODO: find usage.
    """
    cut_points: np.ndarray
    cut_direction: Tuple[float, float, float]


@dataclass
class CalcGradientAlongVectorInput:
    """
    Unknown
    TODO: find usage.
    """
    field: np.ndarray
    field_coords: np.ndarray
    target_endcoding_function: str


# Generated for calc_potential_levels
@dataclass
class CombinedMesh:
    stream_function: List[float]


@dataclass
class InputParameters:
    levels: int
    pot_offset_factor: float
    level_set_method: str
