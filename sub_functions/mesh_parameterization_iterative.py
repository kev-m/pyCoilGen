# System imports
import numpy as np
from scipy.sparse import coo_matrix

# Logging
import logging

# Local imports
from sub_functions.data_structures import DataStructure, Mesh

log = logging.getLogger(__name__)


def mesh_parameterization_iterative(input_mesh : Mesh):
    """
    Performs iterative mesh parameterization based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes".

    Args:
        mesh_in: Input mesh as a DataStructure object containing 'vertices' and 'faces' attributes.

    Returns:
        mesh: Parameterized mesh as a DataStructure object containing 'v', 'n', 'u', 'f', 'e', 'bounds', 'version', 'vidx', and 'fidx' attributes.
    """

    # Initialize mesh properties
    mesh_vertices = input_mesh.get_vertices()
    mesh_faces = input_mesh.get_faces()
    mesh = DataStructure(
        v=mesh_vertices,  # Transpose vertices for column-wise storage
        n=[],
        u=[],
        f=mesh_faces,  # Transpose faces for column-wise storage
        e=[],
        bounds=[np.min(mesh_vertices), np.max(mesh_vertices),
                0.5 * (np.min(mesh_vertices) + np.max(mesh_vertices))],
        version=1,
        vidx=np.arange(1, mesh_vertices.shape[0] + 1),
        fidx=np.arange(1, mesh_faces.shape[0] + 1),
        fn=input_mesh.fn
    )

    # Compute face normals
    for iii in range(mesh.f.shape[0]):
        fvv = mesh.v[mesh.f[iii, :], :]
        log.debug(" fvv: %s", fvv)
        ee1 = fvv[1, :] - fvv[0, :]
        ee2 = fvv[2, :] - fvv[0, :]
        M = np.cross(ee1, ee2)
        log.debug(" M: %s", M)
        mag = np.array([np.sum(M * M, axis=0)])
        log.debug(" mag: %s", mag)
        z = np.where(mag < np.finfo(float).eps)[0]
        mag[z] = 1
        Mlen = np.sqrt(mag)
        log.debug(" Mlen: %s", Mlen)
        temp2 = np.tile(Mlen, (1, M.shape[1]))
        temp = M / temp2
        mesh.fn[iii, :] = temp

    e = np.zeros((mesh.f.shape[0] * 3 * 2, 3))

    # Compute edge list
    for iii in range(mesh.f.shape[0]):
        for jjj in range(3):
            t1 = [mesh.f[iii, jjj], mesh.f[iii, (jjj + 1) % 3], 1]
            t2 = [mesh.f[iii, (jjj + 1) % 3], mesh.f[iii, jjj], 1]
            e[iii * 6 + jjj * 2, :] = t1
            e[iii * 6 + jjj * 2 + 1, :] = t2

    # Update mesh edge attribute
    mesh.e = e

    # Convert edge list to sparse matrix
    mesh.e = coo_matrix((e[:, 3], (e[:, 0] - 1, e[:, 1] - 1)),
                        shape=(mesh.v.shape[0], mesh.v.shape[0]))

    # Find boundary vertices
    iiii, jjjj = np.where(mesh.e == 1)
    mesh.isboundaryv = np.zeros((mesh.v.shape[0],))
    mesh.isboundaryv[iiii] = 1

    # Find boundary edges
    be = np.sort(np.hstack((iiii.reshape(-1, 1), jjjj.reshape(-1, 1))), axis=1)
    be = np.unique(be, axis=0)

    # Determine boundary faces
    mesh.isboundaryf = np.sum(
        mesh.isboundaryv[mesh.f[:, [0, 1, 2]]], axis=1) > 0

    # Initialize variables for boundary loops
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

    if len(bloop) > 0:
        loops.append(bloop)
        loopk += 1

    for kkkk in range(len(loops)):
        loop_sort = loops[kkkk]
        prev_idx = [2, 0, 1]
        loop1, loop2 = loop_sort[0], loop_sort[1]
        ffi, fj = np.where(mesh.f == loop1)

        for iii in range(len(ffi)):
            jp = prev_idx[fj[iii]]

            if mesh.f[ffi[iii], jp] == loop2:
                nL = len(loop_sort)
                loop_sort = loop_sort[nL::-1]

        loops[kkkk] = loop_sort

    if len(loops) > 0:
        loopsize = [len(loop) for loop in loops]
        idx = np.argsort(loopsize)[::-1]

        for kkkk in range(len(idx)):
            mesh.loops[kkkk] = loops[idx[kkkk]]

    else:
        mesh.loops = []

    mesh.te = mesh.e
    mesh.te[mesh.e != 0] = 0

    for ti in range(len(mesh.f)):
        for kkkk in range(3):
            vv1 = mesh.f[ti, kkkk]
            vv2 = mesh.f[ti, (kkkk + 1) % 3]

            if mesh.te[vv1, vv2] != 0:
                ttmp = vv1
                vv1 = vv2
                vv2 = ttmp

            mesh.te[vv1, vv2] = ti

    mesh.valence = np.zeros(len(mesh.v))

    for vi in range(len(mesh.v)):
        jj = np.where(mesh.e[vi])[0]
        mesh.valence[vi] = len(jj)

    mesh.unique_vert_inds = mesh_in.unique_vert_inds
    mesh.n = vertexNormal(triangulation(mesh_faces, mesh_vertices))
    mesh.fn = faceNormal(triangulation(mesh_faces, mesh_vertices))
    iboundary = mesh.vidx[mesh.isboundaryv != 0]
    dists = vmag2(vadd(mesh.v[iboundary], -mesh.v[iboundary[0]]))
    maxi = np.argmax(dists)
    ifixed = [iboundary[0], iboundary[maxi]]
    fixedUV = [[iboundary[0], 0, 0], [iboundary[maxi], 1, 0]]
    N = len(mesh.vidx)
    W = cotanWeights(mesh)
    W = -W
    W[np.arange(N), np.arange(N)] = -np.sum(W, axis=1)
    Ld = np.block([[W, np.zeros((N, N))], [np.zeros((N, N)), W]])
    rhs = np.zeros(2 * N)
    A = sparse.coo_matrix((2 * N, 2 * N), dtype=float)

    for li in range(len(mesh.loops)):
        loop = mesh.loops[li]

        for ii in range(len(loop)):
            jx = loop[ii]
            jy = jx + N
            kx = loop[(ii + 1) % len(loop)]
            ky = kx + N
            A[jx, ky] += 1
            A[kx, jy] -= 1

    A = A + A.T
    Lc = Ld - A
    LcCons = Lc.copy()
    ifixed = np.concatenate((ifixed, ifixed + N))
    LcCons[ifixed, :] = 0
    LcCons[np.diag_indices_from(Lc)] = 1
    rhs[fixedUV[:, 0]] = fixedUV[:, 1]
    rhs[fixedUV[:, 0] + N] = fixedUV[:, 2]
    rhsadd = np.zeros_like(rhs)

    for k in range(len(ifixed)):
        ci = ifixed[k]
        col = LcCons[:, ci]
        col[ci] = 0
        rhsadd += rhs[ci] * col

    LcCons[:, ifixed] = 0
    LcCons[np.diag_indices_from(Lc)] = 1
    rhs -= rhsadd
    mesh.uv = np.linalg.solve(LcCons, rhs)
    mesh.uv = np.column_stack((mesh.uv[:N], mesh.uv[N:2 * N]))

    mesh.vertices = mesh.v.T
    mesh.faces = mesh.f.T
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
    return np.nonzero(mesh.e[nVertex, :] != 0)[0]


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
    W = mesh.e[vertices, :]
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
            faces = [f for f in faces if f != 0]
            verts = np.zeros(len(faces))
            vertfaces = np.zeros(len(faces))

            for kk in range(len(faces)):
                f = mesh.f[faces[kk], :]
                verts[kk] = f[np.logical_and(f != qi, f != qj)]
                vertfaces[kk] = faces[kk]

            sumAB = 0

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
