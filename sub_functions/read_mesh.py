import numpy as np

from build_cylinder_mesh import build_cylinder_mesh
#from build_double_cone_mesh import build_double_cone_mesh
from build_planar_mesh import build_planar_mesh
#from build_circular_mesh import build_circular_mesh
from build_biplanar_mesh import build_biplanar_mesh

def read_mesh(input):
    """
    Read the input mesh and return the coil, target, and shielded meshes.

    Args:
        input (object): Input parameters for reading the mesh.

    Returns:
        coil_mesh (object): Coil mesh object.
        target_mesh (object): Target mesh object.
        shielded_mesh (object): Shielded mesh object.
    """

    coil_mesh = None
    
    # Read the input mesh
    if input.sf_source_file == 'none':

        if input.coil_mesh_file.endswith('.stl'):
            # Load the stl file; read the coil mesh surface
            coil_mesh = stlread_local(input.geometry_source_path + '/' + input.coil_mesh_file)
            coil_mesh = create_unique_noded_mesh(coil_mesh)
            coil_mesh.vertices = coil_mesh.vertices.T
            coil_mesh.faces = coil_mesh.faces.T

        elif input.coil_mesh_file == 'create cylinder mesh':
            # No external mesh is specified by stl file; create default cylindrical mesh
            coil_mesh = build_cylinder_mesh(*input.cylinder_mesh_parameter_list)
            coil_mesh = create_unique_noded_mesh(coil_mesh)
            coil_mesh.vertices = coil_mesh.vertices.T
            coil_mesh.faces = coil_mesh.faces.T

        elif input.coil_mesh_file == 'create double cone mesh':
            # No external mesh is specified by stl file; create default double cone mesh
            coil_mesh = build_double_cone_mesh(*input.double_cone_mesh_parameter_list)
            coil_mesh = create_unique_noded_mesh(coil_mesh)
            coil_mesh.vertices = coil_mesh.vertices.T
            coil_mesh.faces = coil_mesh.faces.T

        elif input.coil_mesh_file == 'create planary mesh':
            # No external mesh is specified by stl file; create default planar mesh
            coil_mesh = build_planar_mesh(*input.planar_mesh_parameter_list)
            coil_mesh = create_unique_noded_mesh(coil_mesh)
            coil_mesh.vertices = coil_mesh.vertices.T
            coil_mesh.faces = coil_mesh.faces.T

        elif input.coil_mesh_file == 'create circular mesh':
            # No external mesh is specified by stl file; create default circular mesh
            coil_mesh = build_circular_mesh(*input.circular_mesh_parameter_list)
            coil_mesh = create_unique_noded_mesh(coil_mesh)
            coil_mesh.vertices = coil_mesh.vertices.T
            coil_mesh.faces = coil_mesh.faces.T

        elif input.coil_mesh_file == 'create bi-planary mesh':
            # No external mesh is specified by stl file; create default biplanar mesh
            coil_mesh = build_biplanar_mesh(*input.biplanar_mesh_parameter_list)
            coil_mesh = create_unique_noded_mesh(coil_mesh)
            coil_mesh.vertices = coil_mesh.vertices.T
            coil_mesh.faces = coil_mesh.faces.T

    else:
        loaded_file = load(cd() + '/' + input.sf_source_file)
        coil_mesh = loaded_file.coil_mesh

    # Read the target mesh surface
    if input.target_mesh_file != 'none':
        target_mesh = stlread_local(input.geometry_source_path + '/' + input.target_mesh_file)
        target_mesh = create_unique_noded_mesh(target_mesh)
    else:
        target_mesh = None

    # Read the shielded mesh surface
    if input.secondary_target_mesh_file != 'none':
        shielded_mesh = stlread_local(input.geometry_source_path + '/' + input.secondary_target_mesh_file)
        shielded_mesh = create_unique_noded_mesh(shielded_mesh)
    else:
        shielded_mesh = None

    return coil_mesh, target_mesh, shielded_mesh


def create_unique_noded_mesh(non_unique_mesh):
    """
    Create a mesh with unique nodes.

    Args:
        non_unique_mesh (object): Mesh object with non-unique nodes.

    Returns:
        unique_noded_mesh (object): Mesh object with unique nodes.
    """

    faces = non_unique_mesh.faces
    verts = non_unique_mesh.vertices

    if faces.shape[1] != 3:
        faces = faces.T

    if verts.shape[1] != 3:
        verts = verts.T

    # Remove double nodes
    unique_verts, unique_inds = np.unique(verts, axis=0, return_inverse=True)
    unique_assignments = np.column_stack((np.arange(1, len(unique_inds) + 1), unique_inds + 1))

    for tri_ind in range(faces.shape[0]):
        faces[tri_ind, 0] = unique_assignments[faces[tri_ind, 0] - 1, 1]
        faces[tri_ind, 1] = unique_assignments[faces[tri_ind, 1] - 1, 1]
        faces[tri_ind, 2] = unique_assignments[faces[tri_ind, 2] - 1, 1]

    unique_noded_mesh = {'faces': faces, 'vertices': unique_verts}

    return unique_noded_mesh


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
    output = {'faces': f, 'vertices': v, 'normals': n}

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
    F = np.empty((numFaces, 3))
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