# System imports
import numpy as np
from scipy.sparse import coo_matrix, find, hstack, vstack, linalg

# Logging
import logging

# Local imports
from sub_functions.data_structures import DataStructure, Mesh

log = logging.getLogger(__name__)

# DEBUG
from helpers.visualisation import compare, compare_contains

def compare_sparse(instance1, instance2, magic=0):
    if not instance1.shape == instance2.shape:
        log.error(" Not the same shape: %s is not %s", np.shape(instance1), np.shape(instance2))
        return False

    num_rows = instance1.shape[0]
    if magic == 0:
        for row_idx in range(num_rows):
            row1 = instance1.getrow(row_idx).data[0]
            row2 = instance2.getrow(row_idx).data

            if len(row1) != len(row2):
                log.error(" Not the same shape at index %d: %s is not %s", row_idx, np.shape(row1), np.shape(row2))
                return False

            if np.allclose(row1, row2) == False:
                log.error(" Not the same value at index [%d]:\n %s ... is not\n %s ...",
                            row_idx, row1[:5], row2[:5])
                return False
        return True
    if magic == 1:
        for row_idx in range(num_rows):
            row1 = instance1.getrow(row_idx).data
            row2 = instance2.getrow(row_idx).data

            if len(row1) != len(row2):
                log.error(" Not the same shape at index %d: %s is not %s", row_idx, np.shape(row1), np.shape(row2))
                return False
            
            if len(row1[0]) == 0:
                break

            if np.allclose(row1, row2) == False:
                log.error(" Not the same value at index [%d]:\n %s ... is not\n %s ...",
                            row_idx, row1[:5], row2[:5])
                return False
        return True
    if magic == 2:
        for row_idx in range(num_rows):
            row1 = instance1.getrow(row_idx).data[0]
            row2 = instance2.getrow(row_idx).data[0]

            if len(row1) != len(row2):
                log.error(" Not the same shape at index %d: %s is not %s", row_idx, np.shape(row1), np.shape(row2))
                return False

            if np.allclose(row1, row2) == False:
                log.error(" Not the same value at index [%d]:\n %s ... is not\n %s ...",
                            row_idx, row1[:5], row2[:5])
                return False
        return True

    return False

def mesh_parameterization_iterative(input_mesh : Mesh, matlab_data = None):
    """
    Performs iterative mesh parameterization based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes".

    Args:
        mesh_in: Input mesh as a DataStructure object containing 'vertices' and 'faces' attributes.

    Returns:
        mesh: Parameterized mesh as a DataStructure object containing 'v', 'n', 'u', 'f', 'e', 'bounds', 'version', 'vidx', and 'fidx' attributes.
    """

    # DEBUG
    if matlab_data is not None:
        coil_part = matlab_data.coil_parts
        m_mesh = coil_part.coil_mesh
        m_debug = m_mesh.mesh_parameterization_iterative
    else:
        m_debug = None

    # Initialize mesh properties
    mesh_vertices = input_mesh.get_vertices()
    mesh_faces = input_mesh.get_faces()
    mesh = DataStructure(
        v=mesh_vertices,
        n=[],
        u=[],
        f=mesh_faces,
        e=[],
        bounds=[np.min(mesh_vertices, axis=0), np.max(mesh_vertices, axis=0),
                0.5 * (np.min(mesh_vertices, axis=0) + np.max(mesh_vertices, axis=0))],
        version=1,
        vidx=np.arange(0, mesh_vertices.shape[0]),
        fidx=np.arange(0, mesh_faces.shape[0]),
        fn=None
    )

    # DEBUG
    if matlab_data is not None and False:
        assert compare(mesh.f, m_mesh.f-1) # Pass
        assert np.allclose(mesh.bounds, m_mesh.bounds) # Pass

    """
    Computed value is never used.
    Using Mesh.face_normals()

    # Compute face normals
    for iii in range(mesh.f.shape[0]):
        fvv = mesh.v[mesh.f[iii, :], :]
        ## log.debug(" fvv: %s", fvv)
        ee1 = fvv[:,1] - fvv[:,0]
        ee2 = fvv[:,2] - fvv[:, 0]
        M = np.cross(ee1, ee2)
        ## log.debug(" M: %s", M)
        mag = np.array([np.sum(M * M, axis=0)])
        ## log.debug(" mag: %s", mag)
        z = np.where(mag < np.finfo(float).eps)[0]
        mag[z] = 1
        Mlen = np.sqrt(mag)
        ## log.debug(" Mlen: %s", Mlen)
        temp2 = np.tile(Mlen, (1, M.shape[0]))
        temp = M / temp2
        mesh.fn[iii] = temp
    """

    # Assuming 'mesh' is a dictionary-like structure containing 'f' and 'v' arrays
    # mesh.f: Faces array (each row represents a face with vertex indices)
    # mesh.v: Vertices array (each row represents a vertex)

    num_faces = mesh.f.shape[0]
    e = np.zeros((num_faces * 3 * 2, 3), dtype=int)

    for iii in range(num_faces):
        for jjj in range(3):
            t1 = [mesh.f[iii, jjj], mesh.f[iii, (jjj + 1) % 3], 1]
            t2 = [mesh.f[iii, (jjj + 1) % 3], mesh.f[iii, jjj], 1]
            e[((iii - 1) * 6) + (jjj * 2) + 1, :] = t1
            e[((iii - 1) * 6) + (jjj * 2) + 2, :] = t2

    # Create a sparse matrix from the 'e' array using numpy's coo_matrix
    mesh.e = coo_matrix((e[:, 2], (e[:, 0], e[:, 1])), shape=(mesh.v.shape[0], mesh.v.shape[0]), dtype=int).tolil()

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(mesh.e, m_mesh.e.astype(int)) # Pass

    # Find boundary vertices
    # mesh.v: Vertices array (each row represents a vertex)
    # mesh.f: Faces array (each row represents a face with vertex indices)

    # Find non-zero entries in the sparse matrix
    iiii, jjjj, _ = find(mesh.e == 1)
    # jjjj, iiii = np.nonzero(mesh.e == 1)

    # Initialize mesh.isboundaryv and set boundary vertices to 1
    mesh.isboundaryv = np.zeros(mesh.v.shape[0], dtype=int)
    mesh.isboundaryv[iiii] = 1

    # Create and sort the boundary edges array 'be'
    be = np.sort(np.hstack((iiii.reshape(-1, 1), jjjj.reshape(-1, 1))), axis=1)
    be = np.unique(be, axis=0)

    # Compute mesh.isboundaryf using the boundary vertex information
    mesh.isboundaryf = (mesh.isboundaryv[mesh.f[:, 0]] + mesh.isboundaryv[mesh.f[:, 1]] + mesh.isboundaryv[mesh.f[:, 2]])
    mesh.isboundaryf = mesh.isboundaryf > 0

    # DEBUG
    if matlab_data is not None and False:
        assert compare(mesh.isboundaryf, m_mesh.isboundaryf) # Pass


    # Initialize variables for loops
    loops = []
    loopk = 1
    bloop = []

    while be.size > 0:
        bloop = []
        a1, a2 = be[0]
        be = np.delete(be, 0, axis=0)
        bloop.append(a2)

        while be.shape[0] > 0:
            nextrow = np.where((be[:, 0] == a2) & (be[:, 1] != a1))[0]

            if nextrow.size > 0:
                b2, b3 = be[nextrow[0], 0], be[nextrow[0], 1]
            else:
                nextrow = np.where((be[:, 1] == a2) & (be[:, 0] != a1))[0]
                b3, b2 = be[nextrow[0], 0], be[nextrow[0], 1]

            if nextrow.size == 0:
                loops.append(bloop)
                loopk += 1
                break
            else:
                be = np.delete(be, nextrow[0], axis=0)
                bloop.append(b3)
                a1, a2 = b2, b3

    if bloop:
        loops.append(bloop)
        loopk += 1

    for kkkk in range(len(loops)):
        loop_sort = loops[kkkk]
        prev_idx = [2, 0, 1]
        loop1 = loop_sort[0]
        loop2 = loop_sort[1]

        ffi, fj = np.where(mesh.f == loop1)

        for iii in range(len(ffi)):
            jp = prev_idx[fj[iii]]

            if mesh.f[ffi[iii], jp] == loop2:
                nL = len(loop_sort)
                loop_sort = loop_sort[nL-1::-1]

        loops[kkkk] = loop_sort

    if loops:
        loopsize = [len(loop) for loop in loops]
        idx = np.argsort(loopsize)[::-1]

        # mesh.loops does not exist yet
        mesh.loops = np.asarray([loops[idx[kkkk]] for kkkk in range(len(idx))])
        #for kkkk in range(len(idx)):
        #    mesh.loops[kkkk] = loops[idx[kkkk]]
    else:
        mesh.loops = []


    # DEBUG
    if matlab_data is not None and False:
        assert compare(mesh.loops.reshape(110,), m_mesh.loops-1) # Pass
        assert compare_sparse(mesh.e, m_mesh.e) # Pass


    mesh.te = mesh.e.copy() # Try 2, mesh.e is already lil
    #mesh.te.data[mesh.e.data != 0] = 0
    #mesh.te = mesh.e.tolil() # To allow index addressing, below
    mesh.te[mesh.e != 0] = 0
    for ti in range(len(mesh.f)):
        for kkkk in range(3):
            vv1 = mesh.f[ti, kkkk]
            vv2 = mesh.f[ti, (kkkk + 1) % 3]

            if mesh.te[vv1, vv2] != 0:
                vv1, vv2 = vv2, vv1

            mesh.te[vv1, vv2] = ti + 1 # NB: Subtract 1 when using values from mesh.te!!

            # DEBUG
            if matlab_data is not None:
                if mesh.te[vv1, vv2] != m_mesh.te[vv1, vv2]:
                    log.debug(" Here!! mesh.te[%d,%d]: %s != %s", vv1,vv2,mesh.te[vv1, vv2],m_mesh.te[vv1, vv2])
                    pass

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(mesh.te, m_mesh.te.astype(int)) # Pass
        pass


    mesh.valence = np.zeros(len(mesh.v))

    for vi in range(len(mesh.v)):
        #ii,jj = np.nonzero(mesh.e[vi])
        _, jj = mesh.e.getrow(vi).nonzero()
        mesh.valence[vi] = len(jj)

    mesh.unique_vert_inds = input_mesh.unique_vert_inds.copy()
    mesh.n = input_mesh.vertex_normals()    # vertexNormal(triangulation(mesh_faces, mesh_vertices))
    mesh.fn = input_mesh.face_normals()     # faceNormal(triangulation(mesh_faces, mesh_vertices))
    iboundary = mesh.vidx[mesh.isboundaryv != 0]

    # DEBUG
    if matlab_data is not None and False:
        assert compare(mesh.valence, m_mesh.valence) # Pass

        assert compare(iboundary, m_debug.iboundary-1) # Pass
        assert compare(mesh.unique_vert_inds, m_mesh.unique_vert_inds-1) # Pass
        assert compare(mesh.fn, m_mesh.fn) # Calculated above
        ## assert compare(mesh.n, m_mesh.n.T) # Fail

    dists = vmag2(vadd(mesh.v[iboundary], -mesh.v[iboundary[0]]))
    maxi = np.argmax(dists)
    ifixed = np.array([iboundary[0], iboundary[maxi]])
    fixedUV = np.array([[iboundary[0], 0, 0], [iboundary[maxi], 1, 0]]) # MATLAB 2,n

    N = len(mesh.vidx)
    W = cotanWeights(mesh) # , m_debug=m_debug)

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(W, m_debug.W1) # Pass

    W = (-W).tolil()
    W[np.arange(N), np.arange(N)] = -np.sum(W, axis=1)

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(W, m_debug.W3) # Pass

    Ld = coo_matrix(vstack([hstack([W, coo_matrix((N, N))]), hstack([coo_matrix((N, N)), W])])).tolil() # Hypothesis...

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(Ld, m_debug.Ld1)


    rhs = np.zeros(2 * N)
    A = coo_matrix((2 * N, 2 * N), dtype=int).tolil()

    for li in range(len(mesh.loops)):
        loop = mesh.loops[li]

        for ii in range(len(loop)):
            jx = loop[ii]
            jy = jx + N
            kx = loop[(ii + 1) % len(loop)]
            ky = kx + N
            A[jx, ky] += 1
            A[kx, jy] -= 1

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(A, m_debug.A1)

    A = A + A.T

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(A.tolil(), m_debug.A2.tolil(), magic=1) # Pass

    Lc = Ld - A
    LcCons = Lc.tolil() # Was Lc.copy(), Using lil because of 'SparseEfficiencyWarning', later
    ifixed = np.concatenate((ifixed, ifixed + N))

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(LcCons, m_debug.LcCons1.tolil(), magic=2) # Pass
        assert compare(ifixed, m_debug.ifixed-1) # Pass

    LcCons[ifixed, :] = 0

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(LcCons.tolil(), m_debug.LcCons2.tolil(), magic=2) # Pass

    # MATLAB: LcCons(sub2ind(size(Lc), ifixed, ifixed)) = 1;
    LcCons[ifixed, ifixed] = 1

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(LcCons.tolil(), m_debug.LcCons3.tolil(), magic=2) # Pass

    rhs[fixedUV[:, 0]] = fixedUV[:, 1]

    # DEBUG
    if matlab_data is not None and False:
        assert compare(rhs, m_debug.rhs1) # Pass

    rhs[fixedUV[:, 0] + N] = fixedUV[:, 2]

    # DEBUG
    if matlab_data is not None and False:
        assert compare(rhs, m_debug.rhs2) # Pass

    rhsadd = np.zeros_like(rhs)

    for k in range(len(ifixed)):
        ci = ifixed[k]
        col = LcCons[:, ci].toarray().flatten()
        col[ci] = 0
        rhsadd += rhs[ci] * col

    # DEBUG
    if matlab_data is not None and False:
        assert compare(rhsadd, m_debug.rhsadd) # Pass

    LcCons[:, ifixed] = 0

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(LcCons.tolil(), m_debug.LcCons4.tolil(), magic=2) # Pass

    LcCons[ifixed, ifixed] = 1

    # DEBUG
    if matlab_data is not None and False:
        assert compare_sparse(LcCons.tolil(), m_debug.LcCons5.tolil(), magic=2) # Pass

    rhs -= rhsadd


    # DEBUG
    if matlab_data is not None and False:
        assert compare(rhs, m_debug.rhs5) # Pass
        assert compare_sparse(LcCons, m_debug.LcCons5.tolil(), magic=2) # Pass

    # mesh.uv = np.linalg.solve(LcCons, rhs)    
    # mesh.uv = np.column_stack((mesh.uv[:N], mesh.uv[N:2 * N]))
    # Solve the sparse linear system using spsolve
    mesh_uv = linalg.spsolve(LcCons, rhs)
    N = len(mesh_uv) // 2
    mesh.uv = np.column_stack((mesh_uv[:N], mesh_uv[N:]))

    # DEBUG
    if matlab_data is not None:
        assert compare(mesh.uv, m_mesh.uv.T)

    #mesh.vertices = mesh.v
    #mesh.faces = mesh.f
    mesh.n = mesh.n.T
    mesh.uv = mesh.uv.T
    mesh.boundary = mesh.loops

    return mesh


def vmag2(M):
    """
    Compute the squared Euclidean norm of each row in M.

    Args:
        M (ndarray): Input matrix.

    Returns:
        ndarray: Squared Euclidean norms of the rows in M.
    """
    return np.sum(M * M, axis=1)


def vadd(M, v):
    """
    Add a vector v to each row of matrix M.

    Args:
        M (ndarray): Input matrix.
        v (ndarray): Vector to be added.

    Returns:
        ndarray: Resulting matrix after adding v to each row of M.
    """
    return M + np.tile(v, (M.shape[0], 1))


def vcot(A, B):
    """
    Compute the cotangent of the angle between two vectors A and B.

    Args:
        A (ndarray): First vector.
        B (ndarray): Second vector.

    Returns:
        float: Cotangent of the angle between A and B.
    """
    tmp = np.dot(A, B)
    cotAB = tmp / np.sqrt(np.dot(A, A) * np.dot(B, B) - tmp * tmp)
    return cotAB


def oneringv(mesh, nVertex):
    """
    Find the one-ring vertices of a given vertex in the mesh.

    Args:
        mesh (Mesh): Mesh object.
        nVertex (int): Index of the vertex.

    Returns:
        ndarray: Indices of the one-ring vertices.
    """
    return np.nonzero(mesh.e[nVertex, :] != 0)[1]


def faceArea(mesh, faces=None):
    """
    Compute the areas of the specified faces or all faces in the mesh.

    Args:
        mesh (Mesh): Mesh object.
        faces (ndarray, optional): Indices of the faces. If not provided, all faces are used.

    Returns:
        ndarray: Face areas.
    """
    if faces is None or len(faces) == 0:
        faces = mesh.fidx

    n = len(faces)
    A = np.zeros(n)

    for i in range(n):
        f = mesh.f[i, :]
        fv = mesh.v[f, :]
        A[i] = triarea(fv[0, :], fv[1, :], fv[2, :])

    return A


def triarea(p1, p2, p3):
    """
    Compute the area of a triangle defined by three points.

    Args:
        p1 (ndarray): First point.
        p2 (ndarray): Second point.
        p3 (ndarray): Third point.

    Returns:
        float: Area of the triangle.
    """
    u = p2 - p1
    v = p3 - p1
    A = 0.5 * np.sqrt(np.dot(u, u) * np.dot(v, v) - np.dot(u, v)**2)
    return A


def cotanWeights(mesh, vertices=None, authalic=False, areaWeighted=False, m_debug=None):
    """
    Compute the cotangent weights for the given vertices in the mesh.

    Args:
        mesh (Mesh): Mesh object.
        vertices (ndarray, optional): Indices of the vertices. If not provided, all vertices are used.
        authalic (bool, optional): Flag indicating whether to use authalic weighting. Default is False.
        areaWeighted (bool, optional): Flag indicating whether to use area-weighted weighting. Default is False.

    Returns:
        ndarray: Cotangent weights for the vertices.
    """
    if vertices is None or len(vertices) == 0:
        vertices = mesh.vidx

    n = len(vertices)
    W = mesh.e[vertices, :].astype(float)
    W[W != 0] = -1

    if areaWeighted:
        faceAreas = faceArea(mesh)
    else:
        faceAreas = np.ones(len(mesh.fidx))

    for i in range(n):
        qi = vertices[i]
        ov = oneringv(mesh, qi)

        for j in range(len(ov)):
            qj = ov[j]
            faces = [mesh.te[qi, qj], mesh.te[qj, qi]]
            # MATLAB is using 0 as some kind of magic number, since MATLAB indexing starts at 1...
            faces = np.array([f for f in faces if f != 0])
            verts = np.zeros(len(faces), dtype=int)
            vertfaces = np.zeros(len(faces), dtype=int)

            for kk in range(len(faces)):
                # Remembering to substract 1 from mesh.te -> faces before using...
                f = mesh.f[faces[kk]-1, :]  # Subtracting 1 from faces before using
                verts[kk] = f[np.logical_and(f != qi, f != qj)]
                vertfaces[kk] = faces[kk]-1 # Subtracting 1 from faces before using

            sumAB = 0.0

            if authalic:
                for kk in range(len(verts)):
                    qo = verts[kk]
                    v1 = mesh.v[qi, :] - mesh.v[qj, :]
                    v2 = mesh.v[qo, :] - mesh.v[qj, :]
                    sumAB += vcot(v1, v2)

                sumAB /= vmag2(mesh.v[qi, :] - mesh.v[qj, :])
            else:
                for kk in range(len(verts)):
                    qo = verts[kk]
                    v1 = mesh.v[qi, :] - mesh.v[qo, :]
                    v2 = mesh.v[qj, :] - mesh.v[qo, :]
                    sumAB += vcot(v1, v2) / faceAreas[int(vertfaces[kk])]

            W[qi, qj] = sumAB

            # DEBUG
            if m_debug is not None:
                if np.allclose(W[qi, qj], m_debug.W1[qi,qj]) == False:
                    log.debug("Here!! W[%d,%d]: %f is not equal to %f", qi, qj, sumAB, m_debug.W1[qi,qj])
                    pass

    return W


def onering(mesh, nVert, mode=None):
    """
    Find the one-ring neighborhood of a vertex in the mesh.

    Args:
        mesh (Mesh): Mesh object.
        nVert (int): Index of the vertex.
        mode (str, optional): Mode for sorting the neighborhood. Can be 'ccw' for counter-clockwise sorting. Default is None.

    Returns:
        ndarray: Array of neighboring triangles.
        ndarray: Array of neighboring vertices.
    """
    vVerts = np.where(mesh.e[nVert, :] != 0)[0]
    vTris = np.concatenate([np.full_like(mesh.te[nVert, vVerts], fill_value=i) for i in range(len(vVerts))] +
                           [np.full_like(mesh.te[vVerts, nVert], fill_value=i) for i in range(len(vVerts))])
    vTris = np.unique(vTris)
    vTris = vTris[vTris != 0]

    if mode is None:
        return vTris, vVerts

    if mode == 'ccw' and len(vVerts) > 0:
        if mesh.isboundaryv[nVert]:
            isb = mesh.isboundaryv[vVerts]
            swapi = np.where(isb != 0)[0]
            tmp = vVerts[0]
            vVerts[0] = vVerts[swapi[0]]
            vVerts[swapi[0]] = tmp

        curv = vVerts[0]
        vSorted = [curv]
        rest = vVerts[1:]
        tnbrs = [mesh.te[nVert, curv], mesh.te[curv, nVert]]
        vnbrs = [0, 0]

        for j in range(2):
            if tnbrs[j] != 0:
                vnbrs[j] = tripick2(mesh.f[tnbrs[j], :], nVert, curv)

        prev = curv
        usev = 0 if vnbrs[0] != 0 else 1
        curv = vnbrs[usev]
        tSorted = [tnbrs[usev]]

        while len(rest) > 0:
            vSorted.append(curv)
            rest = rest[rest != curv]
            tnbrs = [mesh.te[nVert, curv], mesh.te[curv, nVert]]

            if tnbrs[0] == 0 or tnbrs[1] == 0:
                break

            vnbrs = [0, 0]

            for j in range(2):
                if tnbrs[j] != 0:
                    vnbrs[j] = tripick2(mesh.f[tnbrs[j], :], nVert, curv)

            if vnbrs[0] == prev:
                prev = curv
                curv = vnbrs[1]
                tSorted.append(tnbrs[1])
            elif vnbrs[1] == prev:
                prev = curv
                curv = vnbrs[0]
                tSorted.append(tnbrs[0])

        vTris = np.array(tSorted)
        vVerts = np.array(vSorted)

    return vTris, vVerts


def tripick2(face, i, j):
    """
    Find the index of the vertex in the face that is not i or j.

    Args:
        face (ndarray): Array of vertex indices in the face.
        i (int): Index of the first vertex.
        j (int): Index of the second vertex.

    Returns:
        int: Index of the vertex in the face.
    """
    fi = np.where(np.logical_and(face != i, face != j))[0]
    return face[fi]


def vmag(M):
    """
    Compute the Euclidean length (magnitude) of vectors in M.

    Args:
        M (ndarray): Array of vectors.

    Returns:
        ndarray: Array of vector magnitudes.
    """
    return np.sqrt(np.sum(M * M, axis=1))


def ncross(v1, v2):
    """
    Compute the normalized cross product between v1 and v2.

    Args:
        v1 (ndarray): First vector.
        v2 (ndarray): Second vector.

    Returns:
        ndarray: Normalized cross product.
    """
    return  np. linalg. norm(np.cross(v1, v2))
