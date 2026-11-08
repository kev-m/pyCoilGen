# System imports
import numpy as np
from scipy.sparse import coo_matrix, find, hstack, vstack, linalg

# Logging
import logging

# Local imports
from .data_structures import DataStructure, Mesh

log = logging.getLogger(__name__)


def mesh_parameterization_iterative(mesh: Mesh):
    """
    Performs iterative mesh parameterization based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes".

    Args:
        mesh (Mesh): Input mesh.

    Returns:
        mesh: Parameterized mesh as a DataStructure object containing 'v', 'n', 'u', 'f', 'e', 'bounds', 'version', 'vidx', and 'fidx' attributes.
    """
    mesh_vertices = mesh.get_vertices()
    mesh_faces = mesh.get_faces()

    # Initialize mesh properties
    # v,f,fn,n are already initialised
    mesh.bounds = [np.min(mesh_vertices, axis=0), np.max(mesh_vertices, axis=0),
                   0.5 * (np.min(mesh_vertices, axis=0) + np.max(mesh_vertices, axis=0))]

    mesh.version = 1
    mesh.vidx = np.arange(0, mesh_vertices.shape[0])
    mesh.fidx = np.arange(0, mesh_faces.shape[0])

    # The directed edges of all faces, both directions of every side. An edge that two faces share in opposite directions
    # (as for a consistently oriented mesh) is counted twice in the same direction, a boundary edge once.
    corner_a = np.concatenate([mesh.f[:, j] for j in range(3)])
    corner_b = np.concatenate([mesh.f[:, (j + 1) % 3] for j in range(3)])
    rows = np.concatenate((corner_a, corner_b))
    cols = np.concatenate((corner_b, corner_a))

    # Create a sparse matrix from the edges; the duplicate entries are added
    mesh.e = coo_matrix((np.ones(len(rows), dtype=int), (rows, cols)),
                        shape=(mesh.v.shape[0], mesh.v.shape[0]), dtype=int).tolil()

    # Find boundary vertices

    # Find non-zero entries in the sparse matrix
    iiii, jjjj, _ = find(mesh.e == 1)

    # Initialize mesh.isboundaryv and set boundary vertices to 1
    mesh.isboundaryv = np.zeros(mesh.v.shape[0], dtype=int)
    mesh.isboundaryv[iiii] = 1

    # Create and sort the boundary edges array 'be'
    be = np.sort(np.hstack((iiii.reshape(-1, 1), jjjj.reshape(-1, 1))), axis=1)
    be = np.unique(be, axis=0)

    # Compute mesh.isboundaryf using the boundary vertex information
    mesh.isboundaryf = (mesh.isboundaryv[mesh.f[:, 0]] +
                        mesh.isboundaryv[mesh.f[:, 1]] + mesh.isboundaryv[mesh.f[:, 2]])
    mesh.isboundaryf = mesh.isboundaryf > 0

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
    else:
        mesh.loops = []

    # For each directed edge, the (one based) index of the face that has it. A face claims the direction of each of its sides
    # if no earlier face has claimed it, otherwise the opposite direction.
    face_of_directed_edge = {}
    for ti, face in enumerate(mesh.f.tolist()):
        for kkkk in range(3):
            vv1 = face[kkkk]
            vv2 = face[(kkkk + 1) % 3]
            if (vv1, vv2) in face_of_directed_edge:
                vv1, vv2 = vv2, vv1
            face_of_directed_edge[(vv1, vv2)] = ti + 1  # NB: Subtract 1 when using values from mesh.te!!
    te_rows, te_cols = zip(*face_of_directed_edge.keys()) if face_of_directed_edge else ((), ())
    mesh.te = coo_matrix((list(face_of_directed_edge.values()), (te_rows, te_cols)), shape=mesh.e.shape,
                         dtype=mesh.e.dtype).tolil()

    # The valence is the number of neighbours of each vertex
    mesh.valence = np.asarray((mesh.e != 0).sum(axis=1), dtype=float).ravel()

    mesh.unique_vert_inds = mesh.unique_vert_inds.copy()
    iboundary = mesh.vidx[mesh.isboundaryv != 0]

    dists = vmag2(vadd(mesh.v[iboundary], -mesh.v[iboundary[0]]))
    maxi = np.argmax(dists)
    ifixed = np.array([iboundary[0], iboundary[maxi]])
    fixedUV = np.array([[iboundary[0], 0, 0], [iboundary[maxi], 1, 0]])  # MATLAB 2,n

    N = len(mesh.vidx)
    W = cotanWeights(mesh)

    W = (-W).tolil()
    W[np.arange(N), np.arange(N)] = -np.sum(W, axis=1)
    Ld = coo_matrix(vstack([hstack([W, coo_matrix((N, N))]), hstack([coo_matrix((N, N)), W])])).tolil()  # Hypothesis...
    rhs = np.zeros(2 * N)
    # The boundary terms: for every edge of a boundary loop, A[jx, ky] += 1 and A[kx, jy] -= 1 (duplicates are added)
    a_rows, a_cols, a_data = [], [], []
    for loop in mesh.loops:
        jx = np.asarray(loop, dtype=int)
        kx = np.roll(jx, -1)
        a_rows += [jx, kx]
        a_cols += [kx + N, jx + N]
        a_data += [np.ones(len(jx), dtype=int), -np.ones(len(jx), dtype=int)]
    if a_rows:
        A = coo_matrix((np.concatenate(a_data), (np.concatenate(a_rows), np.concatenate(a_cols))),
                       shape=(2 * N, 2 * N), dtype=int).tolil()
    else:
        A = coo_matrix((2 * N, 2 * N), dtype=int).tolil()

    A = A + A.T
    Lc = Ld - A
    LcCons = Lc.tolil()  # Was Lc.copy(), Using lil because of 'SparseEfficiencyWarning', later
    ifixed = np.concatenate((ifixed, ifixed + N))
    LcCons[ifixed, :] = 0
    LcCons[ifixed, ifixed] = 1
    rhs[fixedUV[:, 0]] = fixedUV[:, 1]
    rhs[fixedUV[:, 0] + N] = fixedUV[:, 2]
    rhsadd = np.zeros_like(rhs)

    for k in range(len(ifixed)):
        ci = ifixed[k]
        col = LcCons[:, ci].toarray().flatten()
        col[ci] = 0
        rhsadd += rhs[ci] * col

    LcCons[:, ifixed] = 0
    LcCons[ifixed, ifixed] = 1
    rhs -= rhsadd

    # Solve the sparse linear system using spsolve
    mesh_uv = linalg.spsolve(LcCons, rhs)
    N = len(mesh_uv) // 2
    mesh.uv = np.column_stack((mesh_uv[:N], mesh_uv[N:]))
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


def cotanWeights(mesh, vertices=None, authalic=False, areaWeighted=False):
    """
    Compute the cotangent weights for the given vertices in the mesh.

    All vertices (the default) are evaluated for all edges at once. Other cases use `_cotanWeights_loop`.

    Args:
        mesh (Mesh): Mesh object.
        vertices (ndarray, optional): Indices of the vertices. If not provided, all vertices are used.
        authalic (bool, optional): Flag indicating whether to use authalic weighting. Default is False.
        areaWeighted (bool, optional): Flag indicating whether to use area-weighted weighting. Default is False.

    Returns:
        ndarray: Cotangent weights for the vertices.
    """
    if (vertices is None or len(vertices) == 0) and not authalic:
        weights = _cotanWeights_vectorised(mesh, areaWeighted)
        if weights is not None:
            return weights
    return _cotanWeights_loop(mesh, vertices, authalic, areaWeighted)


def _cotanWeights_vectorised(mesh, areaWeighted=False):
    """
    The cotangent weights of all edges, for all vertices: the sum over the (up to two) faces on the edge of the
    cotangent of the angle at the opposite corner, divided by the face area (if the areas are used).

    Returns:
        sparse matrix, or None if a face on an edge does not have exactly one corner opposite to it (the caller then uses the loop).
    """
    faceAreas = faceArea(mesh) if areaWeighted else np.ones(len(mesh.fidx))
    edges = mesh.e.tocoo()
    edge_present = edges.data != 0
    qi = edges.row[edge_present]
    qj = edges.col[edge_present]
    te = mesh.te.tocsr()

    def cotangent_terms(first, second):
        """For each edge (first -> second): the cot of the angle opposite to it in the face te[first, second], or 0."""
        face = np.asarray(te[first, second]).ravel()
        term = np.zeros(len(face))
        has_face = face != 0
        face_ind = face[has_face] - 1
        corners = mesh.f[face_ind]
        opposite_corner = ~((corners == first[has_face][:, np.newaxis]) | (corners == second[has_face][:, np.newaxis]))
        if not np.all(opposite_corner.sum(axis=1) == 1):
            return None
        opposite = corners[opposite_corner]
        v1 = mesh.v[first[has_face]] - mesh.v[opposite]
        v2 = mesh.v[second[has_face]] - mesh.v[opposite]
        dot = np.sum(v1 * v2, axis=1)
        term[has_face] = dot / np.sqrt(np.sum(v1 * v1, axis=1) * np.sum(v2 * v2, axis=1) - dot * dot) / faceAreas[face_ind]
        return term

    term_a = cotangent_terms(qi, qj)
    term_b = cotangent_terms(qj, qi)
    if term_a is None or term_b is None:
        return None
    return coo_matrix(((0.0 + term_a) + term_b, (qi, qj)), shape=mesh.e.shape).tolil()


def _cotanWeights_loop(mesh, vertices=None, authalic=False, areaWeighted=False):
    """
    The original implementation of `cotanWeights`, edge by edge. It is used for the cases that the vectorised version
    does not handle.

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
                verts[kk] = f[np.logical_and(f != qi, f != qj)].item()  # Exactly one vertex is opposite the edge
                vertfaces[kk] = faces[kk]-1  # Subtracting 1 from faces before using

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
