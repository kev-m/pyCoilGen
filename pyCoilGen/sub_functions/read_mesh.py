# System imports
from argparse import Namespace
import numpy as np
import os
# Logging
import logging

# Local imports
from pyCoilGen.mesh_factory import load_plugins as load_mesh_factory_plugins

from .data_structures import DataStructure, Mesh

log = logging.getLogger(__name__)


def read_mesh(input_args):
    """
    Read the input mesh and return the coil, target, and shielded meshes.

    Args:
        input_args (object): Input parameters for reading the mesh.

    Returns:
        coil_mesh (object): Coil mesh object, or None if no mesh was created (e.g. printing 'help').
        target_mesh (object): Target mesh object.
        shielded_mesh (object): Shielded mesh object.
    """

    coil_mesh = None

    # Read the input mesh
    mesh_plugins = load_mesh_factory_plugins()

    coil_mesh = get_mesh(input_args, 'coil_mesh', 'coil_mesh_file', mesh_plugins)

    if coil_mesh is None:
        return None, None, None

    # Read the target mesh surface
    if input_args.target_mesh_file != 'none':
        target_mesh = Mesh.load_from_file(input_args.geometry_source_path,  input_args.target_mesh_file)
        target_mesh = create_unique_noded_mesh(target_mesh)
    else:
        target_mesh = None

    # Read the shielded mesh surface
    if input_args.secondary_target_mesh_file != 'none':
        shielded_mesh = Mesh.load_from_file(input_args.geometry_source_path, input_args.secondary_target_mesh_file)
        # Removing this, it's not required?
        # shielded_mesh = create_unique_noded_mesh(shielded_mesh)
    else:
        shielded_mesh = None

    return coil_mesh, target_mesh, shielded_mesh


def get_mesh(input_args: Namespace, primary_parameter: str, legacy_parameter: str, mesh_plugins: list):
    """
    Create a mesh using the command-line parameters, with fallback.

    First try the primary/new parameter name, but also support the legacy parameter name.

    Args:
        input_args (Namespace): Input parameters for reading the mesh.
        primary_parameter (str): The name of the primary mesh creation parameter.
        legacy_parameter (str): The name of the legacy mesh creation parameter.
        mesh_plugins (list of modules): The list of modules from `load_mesh_factory_plugins`.

    Returns:
        mesh (Mesh): The created mesh, or None if no mesh was created.

    Raises:
        ValueError if the mesh builder is not found.
    """

    parameter_value = getattr(input_args, primary_parameter)

    if parameter_value == 'none':
        parameter_value = getattr(input_args, legacy_parameter)
        # Preserve legacy behaviour (version 0.x.y)
        log.debug("Using legacy method to load meshes.")
        if parameter_value.endswith('.stl'):
            log.debug("Loading mesh from STL file.")
            # Load the stl file; read the coil mesh surface
            coil_mesh = Mesh.load_from_file(input_args.geometry_source_path,  input_args.coil_mesh_file)
            log.info(" Loaded mesh from STL. Assuming representative normal is [0,0,1]!")
            coil_mesh.normal_rep = np.array([0.0, 0.0, 1.0])
            return coil_mesh
        if parameter_value == 'none':
            return None

    # Version 0.x: Support both 'coil_mesh_file' and 'coil_mesh'. 'coil_mesh' takes priority.
    plugin_name = parameter_value.replace(' ', '_').replace('-', '_')
    print("Using plugin: ", plugin_name)
    if plugin_name == 'help':
        print('Available mesh creators are:')
        for plugin in mesh_plugins:
            name_function = getattr(plugin, 'get_name', None)
            parameters_function = getattr(plugin, 'get_parameters', None)
            if name_function:
                name = name_function()
                if parameters_function:
                    parameters = parameters_function()
                    parameter_name, default_value = parameters[0]
                    print(f"'{name}', Parameter: '{parameter_name}', Default values: {default_value}")
                    for i in range(1, len(parameters)):
                        print(f"\t\tParameter: '{parameter_name}', Default values: {default_value}")

                else:
                    print(f"'{name}', no parameters")
        return None

    found = False
    for plugin in mesh_plugins:
        mesh_creation_function = getattr(plugin, plugin_name, None)
        if mesh_creation_function:
            coil_mesh = mesh_creation_function(input_args)
            found = True
            break

    if found == False:
        raise ValueError(f"No mesh creation method found for {input_args.coil_mesh_file}")


def create_unique_noded_mesh(non_unique_mesh):
    """
    Create a mesh with unique nodes.

    Args:
        non_unique_mesh (DataStructure): Mesh object with non-unique nodes.

    Returns:
        unique_noded_mesh (Mesh): Mesh object with unique nodes.
    """

    faces = non_unique_mesh.faces
    verts = non_unique_mesh.vertices

    mesh = Mesh(vertices=verts, faces=faces)
    # mesh.cleanup() # Changes mesh a lot.
    mesh.normal_rep = non_unique_mesh.normal
    return mesh


def stlread_local(file):
    """
    Read an STL file.

    Args:
        file (str): File path.

    Returns:
        output (object): Mesh object containing faces and vertices.
    """

    if not os.path.isfile(file):
        raise FileNotFoundError(f"File '{file}' not found. If the file is not on the MATLAB's path, "
                                f"be sure to specify the full path to the file.")

    with open(file, 'rb') as fid:
        M = np.fromfile(fid, dtype=np.uint8)

    f, v, n = stlbinary(M)
    # output = {'faces': f, 'vertices': v, 'normals': n}
    output = DataStructure(faces=f, vertices=v, normals=n)
    return output


def stlbinary(M):
    """
    Parse binary STL file data.

    Args:
        M (ndarray): Binary STL file data.

    Returns:
        F (ndarray): Face indices.
        V (ndarray): Vertex coordinates.
        N (ndarray): Face normals.
    """
    F = []
    V = []
    N = []

    if len(M) < 84:
        raise ValueError('Incomplete header information in binary STL file.')

    # Bytes 81-84 are an unsigned 32-bit integer specifying the number of faces that follow.
    numFaces = np.frombuffer(M[80:84], dtype=np.uint32)[0]

    if numFaces == 0:
        print('No data in STL file.')
        return F, V, N

    T = M[84:]
    F = np.empty((numFaces, 3), dtype='int')  # Integer indices
    V = np.empty((3 * numFaces, 3))
    N = np.empty((numFaces, 3))

    numRead = 0
    while numRead < numFaces:
        # Each facet is 50 bytes
        # - Three single precision values specifying the face normal vector
        # - Three single precision values specifying the first vertex (XYZ)
        # - Three single precision values specifying the second vertex (XYZ)
        # - Three single precision values specifying the third vertex (XYZ)
        # - Two unused bytes
        i1 = 50 * numRead
        i2 = i1 + 50
        facet = T[i1:i2]

        n = np.frombuffer(facet[0:12], dtype=np.float32)
        v1 = np.frombuffer(facet[12:24], dtype=np.float32)
        v2 = np.frombuffer(facet[24:36], dtype=np.float32)
        v3 = np.frombuffer(facet[36:48], dtype=np.float32)

        n = np.double(n)
        v = np.double([v1, v2, v3])

        # Figure out where to fit these new vertices, and the face, in the larger F and V collections.
        fInd = numRead
        vInd1 = 3 * fInd
        vInd2 = vInd1 + 3

        V[vInd1:vInd2, :] = v
        F[fInd, :] = np.arange(vInd1, vInd2)
        N[fInd, :] = n

        numRead = numRead + 1

    return F, V, N


def stlascii(M):
    print('ASCII STL files currently not supported.')
    F = []
    V = []
    N = []
    return F, V, N


def isbinary(A):
    if len(A) < 5:
        raise ValueError('File does not appear to be an ASCII or binary STL file.')
    if 'solid' in A[:5].tobytes().decode('utf-8'):
        return False  # ASCII
    else:
        return True  # Binary
