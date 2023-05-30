from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Point:
    uv: np.ndarray
    potential: float

@dataclass
class Loop:
    uv: np.ndarray
    edge_inds: np.ndarray
    current_orientation: int

@dataclass
class UnsortedPoints:
    loop: List[Loop]

@dataclass
class UnarrangedLoops:
    loop: List[Loop]

@dataclass
class RawData:
    unsorted_points: List[UnsortedPoints]
    unarranged_loops: List[UnarrangedLoops]

@dataclass
class ContourLine:
    uv: np.ndarray
    potential: float
    current_orientation: int

@dataclass
class CoilMesh:
    vertices: np.ndarray
    uv: np.ndarray
    faces: np.ndarray

@dataclass
class CoilParts:
    raw: RawData
    contour_lines: List[ContourLine]
    coil_mesh: CoilMesh

@dataclass
class ParameterizedMesh:
    f: np.ndarray
    uv: np.ndarray

@dataclass
class PotentialSortedCutPoints:
    cut_points: np.ndarray
    cut_direction: Tuple[float, float, float]

@dataclass
class GradientData:
    mean_gradient_strength: float
    gradient_out: np.ndarray

@dataclass
class LocalOpeningGab:
    point_1: int
    point_2: int
    opening_gab: float

@dataclass
class CalcLocalOpeningGab2Input:
    loop: Loop
    cut_point: np.ndarray
    cut_direction: Tuple[float, float, float]
    opening_gab: float

@dataclass
class CalcLocalOpeningGabOutput:
    local_opening_gab: float

@dataclass
class CalcGradientAlongVectorOutput:
    mean_gradient_strength: float
    gradient_out: np.ndarray

@dataclass
class Calc3DRotationMatrixOutput:
    rot_mat_out: np.ndarray

@dataclass
class CalcGradientAlongVectorInput:
    field: np.ndarray
    field_coords: np.ndarray
    target_endcoding_function: str

@dataclass
class Calc3DRotationMatrixInput:
    rot_vec: np.ndarray
    rot_angle: float
