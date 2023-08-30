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
from sub_functions.uv_to_xyz import pointLocation

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
            self.trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        else:
            raise ValueError("Either vertices and faces, or trimesh_obj must be provided.")

        # Known properties
        # Assigned in read_mesh
        self.normal_rep = None  # Representative normal for the mesh ([x,y,z])
        # Calculated in parameterize_mesh
        self.v = None           # (n,3) : The array of mesh vertices (n, [x,y,z]).
        self.fn = None          # (n,3) : The face normals (n, [x,y.z]).
        self.n = None           # (n,3) : The vertex normals (n, [x,y.z]).
        self.uv = None          # (n,2) : Vertices, UV texture matrix (n, [x,y,z=0])
        self.boundary = None    # List of 1D lists of vertex indices along mesh boundaries (m,[i]) (parameterize_mesh)

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
        return self.trimesh_obj.face_normals.view(np.ndarray)

    def vertex_normals(self):
        """
        Get the normals of each vertex in the mesh.

        Returns:
            ndarray: An array of vertex normals with shape (num_faces, 3).
        """
        return self.trimesh_obj.vertex_normals.view(np.ndarray)

    def display(self):
        """
        Display the mesh
        """
        return self.trimesh_obj.show()

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

    def boundary_indices(self):
        """
        Get the indices of vertices that are on the boundaries of the mesh.

        For each of n boundaries, return the boundary indices.

        Returns:
            boundaries (ndarray): An (number of boundaries) x (variable) array of boundary face indices.
        """
        # Proposed solution from StackExchange:
        # https://stackoverflow.com/questions/76435070/how-do-i-use-python-trimesh-to-get-boundary-vertex-indices/76907565#76907565
        connections = self.trimesh_obj.edges[trimesh.grouping.group_rows(
            self.trimesh_obj.edges_sorted, require_count=1)]

        # Start with the first vertex and then walk the graph of connections
        if (len(connections) == 0):
            # No boundaries!
            log.error("Mesh has no boundary edges!")
            raise ValueError("Mesh has no boundary edges!")

        # Tweak: Re-order the first entry, lowest first
        connections[0] = sorted(connections[0])

        # Use ChatGPT to reduce connections to lists of connected vertices
        adj_dict = {}
        for conn in connections:
            for vertex in conn:
                adj_dict.setdefault(vertex, []).extend(c for c in conn if c != vertex)

        groups = []
        visited = set()

        def dfs(vertex, group):
            group.append(vertex)
            visited.add(vertex)
            for conn_vertex in adj_dict[vertex]:
                if conn_vertex not in visited:
                    dfs(conn_vertex, group)

        for vertex in adj_dict:
            if vertex in visited:
                continue
            group = []
            dfs(vertex, group)
            groups.append(group)

        # Convert to numpy arrays
        boundaries = np.empty((len(groups)), dtype=object)
        for index, boundary in enumerate(groups):
            # Close the boundary by add the first element to the end
            boundary.append(boundary[0])

            # Swap the order to match MATALB ordering
            if index != len(groups)-1:
                boundary = [node for node in reversed(boundary)] # Reversed, to match MATLAB

            new_array = np.asarray(boundary, dtype=int) 
            boundaries[index] = new_array

        return boundaries

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

    def export(self, file_obj, file_type='stl'):
        """
        Export the current mesh to a file object.
        If file_obj is a filename, file will be written there.

        Supported formats are stl, off, ply, collada, json, dict, glb, dict64, msgpack.

        Args:
            file_obj, One of:
                open writeable file object or 
                file name (str): where to save the mesh
                None: return the export blob
            file_type (str) : Which file type to export as, if `file_name` is passed this is not required.
        """
        self.trimesh_obj.export(file_obj, file_type)

    def get_face_index(self, vertex: np.ndarray, try_harder=True):
        """
        Retrieve the index of the face which contains the provided point.

        If the vertex is on multiple faces, e.g. is an edge, the highest face index is returned.

        Args:
            vertex (ndarray): The point to search for
            try_harder (bool): Whether to also include 2nd order face vertices in the search.

        Returns:
            index (int): The face index or -1 if the point is not within the mesh.
            barycentric (ndarray): The 3 barycentric co-ordinates if the point is within the mesh, else None
        """
        faces = self.get_faces()
        vertices = self.get_vertices()
        diffs = np.abs(vertices - vertex)
        diffs_norm = np.linalg.norm(diffs, axis=1)
        # Find the closest vertices
        min_index = np.argmin(diffs_norm)

        # For each possible vertex, find the faces that reference it.
        possible_face_matches = np.any(faces == min_index, axis=1)

        # Get the indices of the rows that contain the target_face_index
        possible_face_indices = np.where(possible_face_matches)[0]

        # Get the corresponding faces
        faces_to_try = faces[possible_face_indices]

        if try_harder:
            # Add all 2nd degree faces (i.e. faces connected to every vertex)
            more_face_indices = possible_face_indices.tolist()
            for vertex_index in faces_to_try.flatten():
                p2 = np.any(faces == vertex_index, axis=1)
                p2_inds = np.where(p2)[0]
                more_face_indices += p2_inds.tolist()

            possible_face_indices = np.unique(more_face_indices)

            # Get the faces that match these vertices
            faces_to_try = faces[possible_face_indices]

        face_index, barycentric = pointLocation(vertex, faces_to_try, vertices)
        if face_index is not None:
            return possible_face_indices[face_index], barycentric

        # log.debug("get_face_index(%s), No found face", vertex)
        return -1, None


# Helper functions
def append_uv(uv_container, uv_value):
    """
    Append an (n,2) array or item to the uv_container's uv member.
    """
    if uv_container.uv is None:
        uv_container.uv = np.zeros((1, 2), dtype=float)
        uv_container.uv[0] = uv_value
        return
    uv_container.uv = np.vstack((uv_container.uv, [uv_value]))


def append_v(v_container, v_value):
    """
    Append an (n,3) array or item to the uv_container's uv member.
    """
    if v_container.v is None:
        v_container.v = np.zeros((1, 3), dtype=float)
        v_container.v[0] = v_value
        return
    v_container.v = np.vstack((v_container.v, [v_value]))


def append_uv_matlab(uv_container, uv_value: np.ndarray):
    """
    Append a (2,n) array or item to the uv_container's uv member.
    """
    if uv_container.uv is None:
        uv_container.uv = uv_value.copy()
        # if uv_value.shape[1] == 1:
        #    uv_container.uv = np.zeros((2, 1), dtype=float)
        #    uv_container.uv[:,0] = uv_value[:,0]
        # else:
        #    uv_container.uv = uv_value
        return
    # uv_container.uv = np.hstack((uv_container.uv, [uv_value]))
    uv_container.uv = np.hstack((uv_container.uv, uv_value))


def append_v_matlab(v_container, v_value: np.ndarray):
    """
    Append a (3,n) array or item to the v_container's v member.
    """
    if v_container.v is None:
        v_container.v = v_value.copy()
        # if v_value.shape[1] == 1:
        #    v_container.v = np.zeros((3, 1), dtype=float)
        #    v_container.v[:,0] = v_value[:,0]
        # else:
        #    v_container.v = v_value
        return
    # v_container.v = np.hstack((v_container.v, [v_value]))
    v_container.v = np.hstack((v_container.v, v_value))


# Generated for calculate_basis_functions
class BasisElement:
    stream_function_potential: float
    triangles: List[int]                # node_triangles x ?
    one_ring: np.ndarray                # node_triangles x 1
    area: np.ndarray                    # node_triangles x 1
    face_normal: np.ndarray             # node_triangles x 3
    triangle_points_ABC: np.ndarray     # node_triangles x 3 (index, [node_point, point_b, point_c].T) MATLAB shape
    current: np.ndarray                 # node_triangles x 3


@dataclass
class Shape2D:
    uv: np.ndarray = None  # 2D co-ordinates of the shape (2,n)

    def add_uv(self, uv):
        append_uv_matlab(self, uv)


@dataclass
class UnarrangedLoop(Shape2D):
    """
    Represents an unarranged loop in the coil mesh.

    Used by calc_contours_by_triangular_potential_cuts
    """
    edge_inds: List[int] = None
    current_orientation: float = None

    def add_edge(self, edge):
        self.edge_inds.append(edge)

    def add_uv(self, uv):
        append_uv(self, uv)


@dataclass
class UnarrangedLoopContainer:
    loop: List[UnarrangedLoop] = None


# Used in topological_loop_grouping
@dataclass
class UnsortedPoints(Shape2D):
    """
    Represents unsorted contours in the coil mesh.

    Used by calc_contours_by_triangular_potential_cuts
    """
    potential: float = None
    edge_ind: np.ndarray = None


@dataclass
class RawPart:
    """
    Represents the unprocessed collection of points.

    Used by calc_contours_by_triangular_potential_cuts
    """
    unsorted_points: List[UnsortedPoints] = None
    unarranged_loops: List[UnarrangedLoop] = None


@dataclass
class Shape3D(Shape2D):  # Used in topological_loop_grouping
    v: np.ndarray = None  # 3D co-ordinates of the shape (3,n)

    def add_v(self, v):
        append_v_matlab(self, v)

    def copy(self):
        return Shape3D(uv=self.uv.copy(), v=self.v.copy())


@dataclass
class ContourLine(Shape3D):
    """
    Represents a contour line

    Used by calc_contours_by_triangular_potential_cuts
    """
    # v: np.ndarray = None   # 3D co-ordinates of the contour (process_raw_loops) (3,m)
    # uv: np.ndarray = None  # 2D co-ordinates of the contour (process_raw_loops) (2,m)
    potential: float = None  # Potential value of the contour
    current_orientation: int = None


@dataclass
class TopoGroup:                        # CoilPart.groups
    loops: List[ContourLine] = None     # Assigned in topological_loop_grouping
    cutshape: List[Shape2D] = None      # Assigned in interconnect_within_groups
    opened_loop: List[Shape3D] = None   # Assigned in interconnect_within_groups


@dataclass
class ConnectedGroup(Shape3D):          # CoilPart.connected_group
    # uv: np.ndarray = None             # 2D shape (2,n) Assigned in interconnect_within_groups
    # v: np.ndarray = None              # 3D shape (3,n) Assigned in interconnect_within_groups
    return_path: Shape3D = None         # Assigned in interconnect_within_groups
    spiral_in: Shape3D = None           # Assigned in interconnect_within_groups
    spiral_out: Shape3D = None          # Assigned in interconnect_within_groups
    unrolled_coords: np.ndarray = None  # 3D shape (3,n) Assigned in interconnect_among_groups


@dataclass
class Cuts():
    cut1: np.ndarray = None            # Cut data (interconnect_among_groups)
    cut2: np.ndarray = None            # Cut data (interconnect_among_groups)


@dataclass
class Polygon():              # Placeholder class (generate_cylindrical_pcb_print) (2,n)
    data: np.ndarray = None


@dataclass
class PCBPart(Shape2D):
    ind1: np.ndarray = None             # (generate_cylindrical_pcb_print) (2,n)
    ind2: np.ndarray = None             # (generate_cylindrical_pcb_print) (2,n)
    track_shape: np.ndarray = None      # (generate_cylindrical_pcb_print) (2,n)
    polygon_track: Polygon = None       # (generate_cylindrical_pcb_print) (2,n)


@dataclass
class GroupLayout():
    wire_parts: List[PCBPart] = None    # (generate_cylindrical_pcb_print)


@dataclass
class PCBLayer():
    group_layouts: GroupLayout = None


@dataclass
class PCBTrack():
    upper_layer: PCBLayer = None        # (generate_cylindrical_pcb_print)
    lower_layer: PCBLayer = None        # (generate_cylindrical_pcb_print)


@dataclass
class CoilPart:
    coil_mesh: Mesh = None
    one_ring_list: np.ndarray = None        # (calculate_one_ring_by_mesh) (num_vertices,variable) Python shape
    node_triangles: np.ndarray = None       # (calculate_one_ring_by_mesh) (num_vertices,variable)
    node_triangle_mat: np.ndarray = None    # Integer (calculate_one_ring_by_mesh) (num_vertices,num_faces)
    basis_elements: List[BasisElement] = None  # (calculate_basis_functions)
    is_real_triangle_mat: np.ndarray = None  # (calculate_basis_functions) (num_vertices, max_triangle_count_per_node)
    # Integer (calculate_basis_functions) (num_vertices,var,3,3) MATLAB shape
    triangle_corner_coord_mat: np.ndarray = None
    current_mat: np.ndarray = None          # (calculate_basis_functions) (num_vertices, max_triangle_count_per_node, 3)
    area_mat: np.ndarray = None             # (calculate_basis_functions) (num_vertices, max_triangle_count_per_node)
    face_normal_mat: np.ndarray = None      # (calculate_basis_functions) (num_vertices, max_triangle_count_per_node, 3)
    current_density_mat: np.ndarray = None  # (calculate_basis_functions) (num_vertices, num_faces, 3)
    sensitivity_matrix: np.ndarray = None   # (calculate_sensitivity_matrix) (3, target field, num basis)
    resistance_matrix: np.ndarray = None    # (calculate_resistance_matrix) (num_vertices, num_vertices)
    current_density: np.ndarray = None      # (stream_function_optimization) (3, n, num_vertices)
    stream_function: np.ndarray = None      # (stream_function_optimization) (?,?)
    raw: RawPart = None                     # (calc_contours_by_triangular_potential_cuts)
    contour_lines: List[ContourLine] = None  # (process_raw_loops)
    potential_level_list: np.ndarray = None  # Placeholder (calc_potential_levels) (???)
    contour_step: float = None              # Placeholder (calc_potential_levels) (???)
    field_by_loops: np.ndarray = None       # Placeholder (evaluate_loop_significance in process_raw_loops)
    combined_loop_field: np.ndarray = None  # Placeholder (evaluate_loop_significance in process_raw_loops) (3,m)
    loop_significance: np.ndarray = None    # Per contour line (evaluate_loop_significance in process_raw_loops) (n)
    combined_loop_length: float = 0.0       # Length of contour lines (process_raw_loops)
    pcb_track_width: float = 0.0            # PCB track width (find_minimal_contour_distance)
    loop_groups: List[int] = None           # Topological groups (topological_loop_grouping)
    group_levels: np.ndarray = None         # ??? (topological_loop_grouping)
    level_positions: np.ndarray = None      # ??? (topological_loop_grouping)
    groups: List[TopoGroup] = None          # Topological groups (topological_loop_grouping)
    group_centers: List[Shape3D] = None     # The centre of each group (calculate_group_centers)
    connected_group: List[ConnectedGroup] = None  # Connected topological groups (interconnect_within_groups)
    opening_cuts_among_groups: List[Cuts] = None  # ??? (interconnect_among_groups)
    wire_path: Shape3D = None               # The shape of the wire track (interconnect_among_groups)
    shift_array: np.ndarray = None          # ??? (shift_return_paths) (,)
    points_to_shift: np.ndarray = None      # Array of which points to shift (shift_return_paths) (m,)
    pcb_tracks: PCBTrack = None             # (generate_cylindrical_pcb_print)
    layout_surface_mesh: Mesh = None        # Layout mesh (create_sweep_along_surface)
    ohmian_resistance: np.ndarray = None    # Surface wire resistance (create_sweep_along_surface)


# Used by define_target_field
@dataclass
class TargetField:
    """
    Used by define_target_field.py
    """
    b: np.ndarray = None       # (3,num vertices)
    coords: np.ndarray = None  # (3,num vertices)
    weights = None              # (num vertices)
    target_field_group_inds = None  # (num vertices)
    target_gradient_dbdxyz = None  # (3,num vertices)

    def __str__(self):
        return as_string(self)


@dataclass
class CoilSolution:
    """
    Represents a high-level CoilGen solution.

    Attributes:
        coil_parts (list): A list of mesh parts that make up the coil surface.
        target_field: The target field associated with the CoilSolution.
    """
    input_args: any = None
    coil_parts: List[CoilPart] = None
    target_field: TargetField = None
    is_suppressed_point: np.ndarray = None
    combined_mesh: DataStructure = None
    sf_b_field: np.ndarray = None
    primary_surface_ind = None


# Used in calculate_gradient
@dataclass
class LayoutGradient:
    """
    Used by calculate_gradient.py
    """
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


@dataclass
class CutPoint(Shape3D):
    """
    Defines .....

    See Shape3D (which is identical, except that uv and v are (2,n) & (3,n)).

    Assigned in find_group_cut_position
    """
    # uv: np.ndarray = None   # 2D co-ordinates of the shape (n,2)
    # v: np.ndarray = None    # 3D co-ordinates of the shape (n,3)
    segment_ind: List[int] = None  # ???

    # Override to preserve Python shape
    def add_uv(self, uv):
        append_uv(self, uv)

    # Override to preserve Python shape
    def add_v(self, v):
        append_v(self, v)


@dataclass
class CutPosition:
    """
    Defines .....

    Assigned in find_group_cut_position
    """
    cut_point: CutPoint = None   # ????
    high_cut: CutPoint = None   # ????
    low_cut: CutPoint = None    # ???


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
