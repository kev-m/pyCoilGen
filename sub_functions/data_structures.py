# System imports
import numpy as np

# Mesh implementation
import trimesh

# Logging
import logging

from dataclasses import dataclass
from typing import List, Tuple

log = logging.getLogger(__name__)

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
            raise ValueError(
                "Either vertices and faces, or trimesh_obj must be provided.")

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
        return self.trimesh_obj.vertices

    def get_faces(self):
        """
        Returns the faces of the mesh.

        Returns:
            ndarray: The int array of face indices into the vertices array.
        """
        return self.trimesh_obj.faces

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

    def separate_into_get_parts(self):
        """
        Split the mesh into parts, if possible.

        Splits the Mesh contains multiple, seperate, parts.

        Returns:
            List[Mesh]: A list of Mesh objects.
        """
        trimesh_parts = self.trimesh_obj.split(only_watertight=False)
        log.debug("Split into %d parts", len(trimesh_parts))

        parts = []
        for part in trimesh_parts:
            parts.append(DataStructure(coil_mesh=Mesh(part)))

        return parts

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
        if len(np.shape(boundary)) == 3:
            log.debug(" boundary: Extracting sub-array")
            boundary = boundary[0]

        return boundary
    
#
#
#  Generated data classes that are not (yet) used.
#
#


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
class UnarrangedLoop:
    """
    Represents an unarranged loop in the coil mesh.
    """
    loop: List[Loop]


@dataclass
class RawPart:
    """
    Represents a raw part in the coil mesh.
    """
    unsorted_points: List[UnsortedPoint]
    unarranged_loops: List[UnarrangedLoop]


@dataclass
class ContourLine:
    """
    Unknown
    TODO: find usage.
    """
    uv: np.ndarray
    potential: float
    current_orientation: int


@dataclass
class CoilMesh:
    """
    Represents the coil mesh.
    """
    vertices: np.ndarray
    uv: np.ndarray
    faces: np.ndarray
    # boundary ??
    # normals ??


@dataclass
class CoilParts:
    """
    Represents the coil parts.
    """
    raw: RawPart
    contour_lines: List[ContourLine]
    coil_mesh: CoilMesh


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


@dataclass
class Calc3DRotationMatrixInput:
    """
    Unknown
    TODO: find usage.
    """
    rot_vec: np.ndarray
    rot_angle: float

# Generated for calc_potential_levels


@dataclass
class CoilPart:
    stream_function: List[float]
    contour_step: float
    potential_level_list: List[float]


@dataclass
class CombinedMesh:
    stream_function: List[float]


@dataclass
class InputParameters:
    levels: int
    pot_offset_factor: float
    level_set_method: str

# Generated for calculate_basis_functions


@dataclass
class BasisElement:
    stream_function_potential: float
    triangles: List[int]
    one_ring: np.ndarray
    area: np.ndarray
    face_normal: np.ndarray
    triangle_points_ABC: np.ndarray
    current: np.ndarray


@dataclass
class CoilPart:
    is_real_triangle_mat: np.ndarray
    triangle_corner_coord_mat: np.ndarray
    current_mat: np.ndarray
    area_mat: np.ndarray
    face_normal_mat: np.ndarray
    basis_elements: List[BasisElement]
    current_density_mat: np.ndarray

# Generated for calculate_gradient


@dataclass
class LayoutGradient:
    dBxdxyz: np.ndarray
    dBydxyz: np.ndarray
    dBzdxyz: np.ndarray
    gradient_in_target_direction: np.ndarray = None
    mean_gradient_in_target_direction: float = None
    std_gradient_in_target_direction: float = None
