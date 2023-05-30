from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

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

  