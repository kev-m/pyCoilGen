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
class PotentialSortedCutPoints:
    """
    Unknown
    TODO: find usage.
    """
    cut_points: np.ndarray
    cut_direction: Tuple[float, float, float]

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
    Unknown
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
class LoopCalculationInput:
    """
    Represents the input data for loop calculation.
    """
    loop: Loop
    point_1: int
    point_2: int
    opening_gab: float

@dataclass
class LocalOpeningGabCalculationInput:
    """
    Represents the input data for local opening gab calculation.
    """
    loop: Loop
    point_1: int
    point_2: int
    opening_gab: float

@dataclass
class CalcLocalOpeningGab2Input:
    """
    Unknown, might be duplicate.
    TODO: find usage.
    """
    loop: Loop
    cut_point: np.ndarray
    cut_direction: Tuple[float, float, float]
    opening_gab: float

@dataclass
class CalcLocalOpeningGabOutput:
    """
    Unknown
    TODO: find usage.
    """
    local_opening_gab: float

@dataclass
class CalcGradientAlongVectorOutput:
    """
    Unknown
    TODO: find usage.
    """
    mean_gradient_strength: float
    gradient_out: np.ndarray

@dataclass
class Calc3DRotationMatrixOutput:
    """
    Unknown
    TODO: find usage.
    """
    rot_mat_out: np.ndarray

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
