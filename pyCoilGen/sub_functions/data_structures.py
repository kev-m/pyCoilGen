# System imports
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
# Mesh implementation
import trimesh

# Logging
import logging

# Local imports
from .constants import *
from .uv_to_xyz import pointLocation, barycentric_to_cartesian
from pyCoilGen.helpers.common import find_file

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
        self.v = self.get_vertices()    # (n,3) : The array of mesh vertices (n, [x,y,z]).
        self.f = self.get_faces()       # (n,3) : The array of face indices (n, [vi1,vi2,vi3]).
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
    def load_from_file(default_path, filename):
        """
        Load a mesh from a file.

        Tries to load the file from the given path locally, in the data directory and in the installed package data
        directory.

        Args:
            filename (str): The path to the mesh file.

        Returns:
            Mesh: An instance of Mesh representing the loaded mesh.

        Raises:
            FileNotFoundError if the file is not found.
        """
        trimesh_obj = trimesh.load_mesh(find_file(default_path, filename))
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
                boundary = [node for node in reversed(boundary)]  # Reversed, to match MATLAB

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
        faces = self.f
        vertices = self.v
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

        return -1, None

    def uv_to_xyz(self, points_in_2d_in: np.ndarray, planary_uv: np.ndarray, num_attempts=1000):
        """
        Convert 2D surface coordinates to 3D xyz coordinates of the 3D coil surface.

        Args:
            points_in_2d (ndarray): The input 2D points with shape (2,n).
            planary_uv (ndarray): Mesh UV array (num vertices,2).
            num_attempts (int) : If the point is not on the mesh, how many times to search nearby points.

        Returns:
            points_out_3d (ndarray): The 3D xyz coordinates of the points with shape (3,n).
            points_out_2d (ndarray): The updated 2D points after removing points that could not be assigned to a triangle, with shape (2,n).
        """
        # NOTE: MATLAB coords: points_in_2d, points_out_3d and points_out_2d
        curved_mesh_vertices = self.v
        curved_mesh_faces = self.f

        planary_mesh = Mesh(faces=curved_mesh_faces, vertices=planary_uv)
        planar_vertices = planary_mesh.trimesh_obj.vertices

        mean_pos = np.mean(planar_vertices, axis=0)
        diameters = np.linalg.norm(planar_vertices - mean_pos, axis=1)
        avg_mesh_diameter = np.mean(diameters)

        points_out_3d = np.zeros((points_in_2d_in.shape[1], 3))  # Python shape
        points_out_2d = points_in_2d_in.T.copy()  # Python shape
        num_deleted_points = 0
        for point_ind in range(points_out_2d.shape[0]):  # MATLAB sgape
            point = points_out_2d[point_ind - num_deleted_points]
            # Find the target triangle and barycentric coordinates of the point on the planar mesh
            # target_triangle, barycentric = get_target_triangle(point, planary_mesh, proximity)
            target_triangle, barycentric = planary_mesh.get_face_index(point)

            attempts = 0
            np.random.seed(3)  # Setting the seed to improve testing robustness
            while target_triangle == -1:
                # If the point is not directly on a triangle, perturb the point slightly and try again
                rand = (0.5 - np.random.rand(2))
                perturbed_point = point + avg_mesh_diameter * np.array([rand[0], rand[1]]) / 1000
                # target_triangle, barycentric = get_target_triangle(perturbed_point, planary_mesh, proximity)
                target_triangle, barycentric = planary_mesh.get_face_index(perturbed_point)
                attempts += 1
                if attempts > num_attempts:
                    log.warning('point %s at index %d can not be assigned to any triangle.', point, point_ind)
                    break

            if target_triangle != -1:
                if attempts > 0:
                    point = perturbed_point
                # Convert the 2D barycentric coordinates to 3D Cartesian coordinates
                face_vertices_3d = curved_mesh_vertices[curved_mesh_faces[target_triangle]]
                points_out_3d[point_ind - num_deleted_points,
                              :] = barycentric_to_cartesian(barycentric, face_vertices_3d)
            else:
                # Remove the point if it cannot be assigned to a triangle
                points_out_2d = np.delete(points_out_2d, point_ind - num_deleted_points, axis=0)  # Python shape
                points_out_3d = np.delete(points_out_3d, point_ind - num_deleted_points, axis=0)  # Python shape
                num_deleted_points += 1

        return points_out_3d.T, points_out_2d.T  # Return as MATLAB shape


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
    current_orientation: float = None
    edge_inds: np.ndarray = None  # Converted from list

    def add_edge(self, edge):
        if self.edge_inds is None:
            self.edge_inds = np.zeros((1, 2), dtype=int)
            self.edge_inds[0] = edge
            return
        self.edge_inds = np.vstack((self.edge_inds, [edge]))

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
    cutshape: List[Shape2D] = None      # 2D Shape (2,n) Assigned in interconnect_within_groups
    opened_loop: List[Shape3D] = None   # 3D Shape (2,n) Assigned in interconnect_within_groups


@dataclass
class ConnectedGroup(Shape3D):          # CoilPart.connected_group
    # uv: np.ndarray = None             # 2D shape (2,n) Assigned in interconnect_within_groups
    # v: np.ndarray = None              # 3D shape (3,n) Assigned in interconnect_within_groups
    return_path: Shape3D = None         # 3D shape (3,n) Assigned in interconnect_within_groups
    spiral_in: Shape3D = None           # 3D shape (3,n) Assigned in interconnect_within_groups
    spiral_out: Shape3D = None          # 3D shape (3,n) Assigned in interconnect_within_groups
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
    stream_function: np.ndarray = None      # (stream_function_optimization) (num_vertices)
    raw: RawPart = None                     # (calc_contours_by_triangular_potential_cuts)
    contour_lines: List[ContourLine] = None  # (process_raw_loops)
    potential_level_list: np.ndarray = None  # Placeholder (calc_potential_levels) (???)
    contour_step: float = None              # Placeholder (calc_potential_levels) (???)
    # Placeholder (evaluate_loop_significance in process_raw_loops and evaluate_field_errors) (3,num_vertices,num contours)
    field_by_loops: np.ndarray = None
    combined_loop_field: np.ndarray = None  # Placeholder (evaluate_loop_significance in process_raw_loops) (3,m)
    loop_significance: np.ndarray = None    # Per contour line (evaluate_loop_significance in process_raw_loops) (n)
    combined_loop_length: float = 0.0       # Length of contour lines (process_raw_loops)
    pcb_track_width: float = 0.0            # PCB track width (find_minimal_contour_distance)
    loop_groups: List[int] = None           # Topological groups (topological_loop_grouping)
    group_levels: np.ndarray = None         # 0-based index of ??? (topological_loop_grouping)
    level_positions: List[List] = None      # 0-based list of indices of ??? (topological_loop_grouping)
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
    field_by_loops2: np.ndarray = None      # (evaluate_field_errors) (3,num_vertices)
    field_by_layout: np.ndarray = None      # (evaluate_field_errors) (3,num_vertices)

    def __repr__():
        return f'CoilPart'


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
class FieldErrors:
    """
    Maximum and mean relative errors, in percent.

    Used by evaluate_field_errors.py
    """
    max_rel_error_layout_vs_target: float = None
    mean_rel_error_layout_vs_target: float = None

    max_rel_error_unconnected_contours_vs_target: float = None
    mean_rel_error_unconnected_contours_vs_target: float = None

    max_rel_error_layout_vs_stream_function_field: float = None
    mean_rel_error_layout_vs_stream_function_field: float = None

    max_rel_error_unconnected_contours_vs_stream_function_field: float = None
    mean_rel_error_unconnected_contours_vs_stream_function_field: float = None


@dataclass
class SolutionErrors:
    """
    Computed target fields for the solution.

    Used by evaluate_field_errors.py
    """
    field_error_vals: FieldErrors = None               # Detailed errors, by field
    combined_field_layout: np.ndarray = None           # Resulting target field by final wire path.
    combined_field_loops: np.ndarray = None            # Resulting target field by contour loops.
    combined_field_layout_per1Amp: np.ndarray = None   # Resulting target field by final wire path, for 1 A current.
    combined_field_loops_per1Amp: np.ndarray = None    # Resulting target field by contours loops, for 1 A current.
    opt_current_layout: float = None                   # The current that will achieve the desired target field.


# Used in calculate_gradient
@dataclass
class LayoutGradient:
    """
    Used by calculate_gradient.py
    """
    dBxdxyz: np.ndarray = None                          # in [mT/m/A]
    dBydxyz: np.ndarray = None                          # in [mT/m/A]
    dBzdxyz: np.ndarray = None                          # in [mT/m/A]
    gradient_in_target_direction: np.ndarray = None     # in [mT/m/A]
    mean_gradient_in_target_direction: float = None     # in [mT/m/A]
    std_gradient_in_target_direction: float = None      # in [mT/m/A]


@dataclass
class CoilSolution:
    """
    Represents a high-level CoilGen solution.

    Attributes:
        coil_parts (list): A list of mesh parts that make up the coil surface.
        target_field: The target field associated with the CoilSolution.
    """
    input_args: any = None                  # Copy of the input parameters.
    coil_parts: List[CoilPart] = None       # Intermediate data for each coil part.
    target_field: TargetField = None        # TargetField data.
    is_suppressed_point: np.ndarray = None  # Array indicating which entries in the TargetField are suppressed.
    combined_mesh: DataStructure = None     # A composite mesh constructed from all CoilParts meshes.
    sf_b_field: np.ndarray = None           # The magnetic field generated by the stream function (n,3)
    primary_surface_ind = None              # Index of the primary surface in the of the coil_parts list.
    solution_errors: SolutionErrors = None  # Computed errors.
    coil_gradient: LayoutGradient = None    # Computed gradient field from the coil solution.


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
