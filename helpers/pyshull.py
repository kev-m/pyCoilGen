# From:
# https://github.com/TimSC/pyshull/blob/master/pyshull.py
# Edited down to remove unused code.
"""
pyshull.py : S-Hull Delaunay triangulation implemented in pure Python. Based on S-hull: a fast sweep-hull routine for
Delaunay triangulation by David Sinclair. This implementation is 10-100 times slower than the reported results of 
S-HULL implemented in C.

earclipping.py : Based on Triangulation by Ear Clipping by David Eberly. Triangularisation of a simple polygon (no self 
intersections) with holes.

Tested on Python 2.7 and 3.4

Copyright (c) 2014-2016, Tim Sheerman-Chase

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the 
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import math


def CalcDist(a, b):
    # Pythagorean theorem
    return ((a[0] - b[0]) ** 2. + (a[1] - b[1]) ** 2.) ** 0.5


def CalcDistCached(pts, a, b, distCache):
    ptIds = (a, b)
    distId = min(ptIds), max(ptIds)
    if distId in distCache:
        return distCache[distId]

    dist = CalcDist(pts[a], pts[b])
    distCache[distId] = dist
    return dist


def RadialDistance(pts, seedIndex):
    dists = []
    seedPt = pts[seedIndex]

    for ptNum, pt in enumerate(pts):
        dist = CalcDist(pt, seedPt)
        dists.append((dist, ptNum))

    dists.sort()
    return dists


def FindSmallestCircumCircle(pts, firstIndex, secondIndex):

    # http://www.mathopenref.com/trianglecircumcircle.html
    a = CalcDist(pts[firstIndex], pts[secondIndex])
    if a == 0.:
        raise Exception("Zero distance between duplicate points is not allowed")

    diams = []
    for ptNum, pt in enumerate(pts):
        if ptNum == firstIndex:
            continue
        if ptNum == secondIndex:
            continue
        b = CalcDist(pts[firstIndex], pts[ptNum])
        c = CalcDist(pts[secondIndex], pts[ptNum])

        # https://en.wikipedia.org/wiki/Heron%27s_formula#Numerical_stability
        x1 = (a+(b+c))
        x2 = (c-(a-b))
        x3 = (c+(a-b))
        x4 = (a+(b-c))
        x = x1*x2*x3*x4
        if x > 0.:
            sqrtx = x**0.5
            if sqrtx > 0.:
                diam = 0.5*a*b*c/sqrtx
                diams.append((diam, ptNum))
            else:
                # Prevent division by zero
                diams.append((float("inf"), ptNum))
        else:
            # Numerical instability detected
            diams.append((float("inf"), ptNum))

    diams.sort()
    return diams


def CircumCircleCentre(pta, ptb, ptc):
    # https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates
    pta2 = (pta[0]**2.+pta[1]**2.)
    ptb2 = (ptb[0]**2.+ptb[1]**2.)
    ptc2 = (ptc[0]**2.+ptc[1]**2.)

    d = 2.*(pta[0]*(ptb[1]-ptc[1])+ptb[0]*(ptc[1]-pta[1])+ptc[0]*(pta[1]-ptb[1]))
    if d == 0.:
        raise RuntimeError("Could not find circumcircle centre")

    ux = (pta2*(ptb[1]-ptc[1]) + ptb2*(ptc[1]-pta[1]) + ptc2*(pta[1]-ptb[1])) / d
    uy = (pta2*(ptc[0]-ptb[0]) + ptb2*(pta[0]-ptc[0]) + ptc2*(ptb[0]-pta[0])) / d

    return ux, uy


def RightHandedCheck(pts, pt1, pt2, pt3):
    vec21 = (pts[pt1][0] - pts[pt2][0], pts[pt1][1] - pts[pt2][1])
    vec23 = (pts[pt3][0] - pts[pt2][0], pts[pt3][1] - pts[pt2][1])
    return -(vec21[0] * vec23[1] - vec21[1] * vec23[0])


def FormTriangles(pts, seedTriangle, orderToAddPts):
    triangles = [seedTriangle]
    hull = seedTriangle[:]

    for ptToAdd in orderToAddPts:
        # Check which hull faces are visible
        visInd = []
        visList = []
        for hInd in range(len(hull)):
            vis = RightHandedCheck(pts, hull[hInd], hull[(hInd+1) % len(hull)], ptToAdd)
            visList.append(vis)
            if vis <= 0.:
                visInd.append(hInd)

        if len(visInd) == 0:
            raise Exception("No hull sides visible")

        # Check for range of sides that are visible
        firstSide = 0
        while firstSide in visInd:
            firstSide += 1
            if firstSide >= len(hull):
                raise Exception("No sides are not visible to point")

        while firstSide not in visInd:
            firstSide = (firstSide + 1) % len(hull)

        lastSide = firstSide
        while (lastSide+1) % len(hull) in visInd:
            lastSide = (lastSide+1) % len(hull)

        # Get copy of retained section of hull
        cursor = (lastSide + 1) % len(hull)
        newHull = []
        iterating = True
        while iterating:
            newHull.append(hull[cursor])
            if cursor in visInd:
                iterating = False
            cursor = (cursor + 1) % len(hull)

        # Add new point to hull
        newHull.append(ptToAdd)

        # Form new triangles
        cursor = firstSide
        iterating = True
        while iterating:
            tri = (hull[cursor], ptToAdd, hull[(cursor+1) % len(hull)])
            # print("Found triangle", tri)
            triangles.append(tri)

            if cursor == lastSide:
                iterating = False
            cursor = (cursor + 1) % len(hull)

        # print("newhull" , newHull)
        hull = newHull
    return hull, triangles


def CalcTriangleAng(pts, angleCache, pt1, pt2, pt3):

    angId = (pt1, pt2, pt3)
    if angId in angleCache:
        return angleCache[angId]

    # Angle is computed on pt3. pt1 and pt2 define the side opposite the angle
    pt1v = pts[pt1]
    pt2v = pts[pt2]
    pt3v = pts[pt3]
    v31 = (pt1v[0] - pt3v[0], pt1v[1] - pt3v[1])
    v32 = (pt2v[0] - pt3v[0], pt2v[1] - pt3v[1])
    mv31 = (v31[0]**2. + v31[1]**2.) ** 0.5
    mv32 = (v32[0]**2. + v32[1]**2.) ** 0.5
    if mv31 == 0. or mv32 == 0.:
        raise RuntimeError("Angle not defined for zero area triangles")

    v31n = [c / mv31 for c in v31]
    v32n = [c / mv32 for c in v32]
    crossProd = - v31n[0] * v32n[1] + v31n[1] * v32n[0]
    dotProd = v31n[0] * v32n[0] + v31n[1] * v32n[1]

    # Limit to valid range
    if dotProd > 1.:
        dotProd = 1.
    if dotProd < -1.:
        dotProd = -1.

    # print(crossProd < 0., crossProd)
    # print(math.asin(crossProd), math.acos(dotProd), cosAng)
    if crossProd < 0.:
        # Reflex angle detected
        trigAng = 2. * math.pi - math.acos(dotProd)
    else:
        # Acute or obtuse angle
        trigAng = math.acos(dotProd)

    angleCache[angId] = trigAng
    return trigAng


def CheckAndFlipTrianglePair(pts, triOrdered1, triOrdered2, angleCache, distCache, debugMode=0):

    if debugMode and RightHandedCheck(pts, *triOrdered1) < 0.:
        raise RuntimeError("Left hand triangle detected", triOrdered1)
    if debugMode and RightHandedCheck(pts, *triOrdered2) < 0.:
        raise RuntimeError("Left hand triangle detected", triOrdered2)
    # print("triOrdered1", triOrdered1)
    # print("triOrdered2", triOrdered2)
    quad = triOrdered1[0], triOrdered1[2], triOrdered2[2], triOrdered2[1]
    # print("quad", quad)

    try:
        t1 = CalcTriangleAng(pts, angleCache, quad[0], quad[2], quad[1])
        t3 = CalcTriangleAng(pts, angleCache, quad[2], quad[0], quad[3])
    except RuntimeError:
        return False, triOrdered1, triOrdered2

    flipDegenerateTri = (t1 == math.pi or t3 == math.pi)
    angTotal = t1 + t3
    flipForDelaunay = angTotal > math.pi

    # print(ang1, ang2, angTotal)
    if flipDegenerateTri or flipForDelaunay:
        # print("Flip possibly required", angTotal, triOrdered1, triOrdered2)
        try:
            t2 = CalcTriangleAng(pts, angleCache, quad[1], quad[3], quad[2])
            t4 = CalcTriangleAng(pts, angleCache, quad[3], quad[1], quad[0])
        except RuntimeError:
            return False, triOrdered1, triOrdered2
        # t1 + t2 + t3 + t4 == 2 * math.pi

        if flipDegenerateTri and (t2 > math.pi or t4 > math.pi):
            # Flipping would create an overlap
            return False, triOrdered1, triOrdered2

        if t2 == math.pi or t4 == math.pi:
            # print(t1, t2, t3, t4)
            # Flipping would create triangle of zero size
            return False, triOrdered1, triOrdered2

        flipTri1 = (triOrdered2[1], triOrdered1[2], triOrdered1[0])
        flipTri2 = (triOrdered1[2], triOrdered2[1], triOrdered1[1])
        # print(flipTri1, flipTri2)
        flipAngTotal = t2 + t4
        # print("Angle when flipped", flipAngTotal)

        if flipForDelaunay and flipAngTotal >= angTotal:
            # print("Abort flip", flipAngTotal)
            # No improvement when flipped, so abort flip
            return False, triOrdered1, triOrdered2

        # print(flipTri1, RightHandedCheck(pts, *flipTri1))
        # print(flipTri2, RightHandedCheck(pts, *flipTri2))

        rhCheck1, rhCheck2 = 0., 0.
        if debugMode:
            rhCheck1 = RightHandedCheck(pts, *flipTri1)
            rhCheck2 = RightHandedCheck(pts, *flipTri2)

        # Ensure they are right handed
        if rhCheck1 < 0.:
            raise RuntimeError("Left hand triangle detected", flipTri1)
        if rhCheck2 < 0.:
            raise RuntimeError("Left hand triangle detected", flipTri2)

        # print("flipped", flipTri1, flipTri2, flipDegenerateTri, flipForDelaunay)
        return True, flipTri1, flipTri2

    return False, triOrdered1, triOrdered2


def HasCommonEdge(tri1, tri2):

    edgeInd1 = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    edgeInd2 = [(2, 1, 0), (1, 0, 2), (0, 2, 1)]
    for ei1 in edgeInd1:
        pt1 = tri1[ei1[0]]
        pt2 = tri1[ei1[1]]
        for ei2 in edgeInd2:
            if pt1 == tri2[ei2[0]] and pt2 == tri2[ei2[1]]:
                return (ei1, ei2)
    return None


def RemoveTriangleFromCommonEdges(sharedEdges, triangles, triNum):

    tri = triangles[triNum]
    edgeInds = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for edgeInd in edgeInds:
        edge = (tri[edgeInd[0]], tri[edgeInd[1]])
        edgeId = min(edge), max(edge)
        sharedEdges[edgeId].remove(triNum)


def AddTriangleToCommonEdges(sharedEdges, triangles, triNum):

    tri = triangles[triNum]
    edgeInds = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for edgeInd in edgeInds:
        edge = (tri[edgeInd[0]], tri[edgeInd[1]])
        edgeId = min(edge), max(edge)
        if edgeId not in sharedEdges:
            sharedEdges[edgeId] = []
        sharedEdges[edgeId].append(triNum)


def FlipTriangles(pts, triangles, nodeOrdering=None):

    # Set all triangle windings the same way
    for triNum, tri in enumerate(triangles):
        rhCheck = RightHandedCheck(pts, *tri)
        if rhCheck < 0:
            triangles[triNum] = tri[::-1]
        if nodeOrdering == None and rhCheck != 0.:
            nodeOrdering = (rhCheck > 0.)

    # Catalog shared edges
    sharedEdges = {}
    for triNum, tri in enumerate(triangles):
        AddTriangleToCommonEdges(sharedEdges, triangles, triNum)

    # print(sharedEdges)
    angleCache = {}
    distCache = {}
    previousConfigurations = [triangles[:]]

    running = True
    while running:
        # Since we are modifying the edge structure, take a static copy of keys
        sharedEdgeKeys = list(sharedEdges.keys())

        count = 0

        for edgeKey in sharedEdgeKeys:
            edge = sharedEdges[edgeKey][:]
            if len(edge) < 2:
                continue

            tri1 = triangles[edge[0]]
            tri2 = triangles[edge[1]]

            commonEdge = HasCommonEdge(tri1, tri2)
            if commonEdge is None:
                # print("err", tri1, tri2)
                raise Exception("Expected common edge")
            triInd1, triInd2 = commonEdge
            # print("original ind", tri1, tri2)

            # Reorder nodes so the common edge is the first two verticies
            triOrdered1 = (tri1[triInd1[0]], tri1[triInd1[1]], tri1[triInd1[2]])  # 1st and 2nd are common edge
            triOrdered2 = (tri2[triInd2[0]], tri2[triInd2[2]], tri2[triInd2[1]])  # 1st and 3rd are common edge
            # print(triOrdered1, triOrdered2)

            # Check if triangle flip is needed
            flipNeeded, ft1, ft2 = CheckAndFlipTrianglePair(pts, triOrdered1, triOrdered2, angleCache, distCache)

            if flipNeeded:
                RemoveTriangleFromCommonEdges(sharedEdges, triangles, edge[0])
                RemoveTriangleFromCommonEdges(sharedEdges, triangles, edge[1])

                triangles[edge[0]] = ft1
                triangles[edge[1]] = ft2

                AddTriangleToCommonEdges(sharedEdges, triangles, edge[0])
                AddTriangleToCommonEdges(sharedEdges, triangles, edge[1])

                count += 1

                if 0:
                    print(tri1, tri2)
                    print([pts[c] for c in tri1], [pts[c] for c in tri2])

                    import matplotlib.pyplot as plt
                    import numpy as np
                    plt.clf()
                    for tn, tri in enumerate(triangles):
                        col = "g--"
                        if tn == edge[0]:
                            col = "r--"
                        if tn == edge[1]:
                            col = "b--"
                        tri2 = list(tri[:])
                        tri2.append(tri[0])
                        ptsArr = np.array(pts)
                        plt.plot(ptsArr[tri2, 0], ptsArr[tri2, 1], col)

                    # plt.plot(pts[:,0], pts[:,1], 'x')
                    plt.show()

        if count > 0 and triangles in previousConfigurations:

            # Prevent an infinite loop of triangle flipping
            exception = RuntimeError("Cannot find delaunay arrangement")
            exception.triangles = triangles
            raise exception

        previousConfigurations.append(triangles[:])

        if count == 0:
            running = False

    if nodeOrdering is False:
        # Reverse order of triangles to match input node order
        for triNum, tri in enumerate(triangles):
            triangles[triNum] = tri[::-1]

    return triangles


def RemoveDuplicatePoints(pts):
    filteredPts = set([tuple(pt) for pt in pts])
    return list(filteredPts)


def HeronsFormula(pts, tri):

    a = CalcDist(pts[tri[0]], pts[tri[1]])
    b = CalcDist(pts[tri[1]], pts[tri[2]])
    c = CalcDist(pts[tri[2]], pts[tri[0]])

    # https://en.wikipedia.org/wiki/Heron%27s_formula#Numerical_stability
    x1 = (a+(b+c))
    x2 = (c-(a-b))
    x3 = (c+(a-b))
    x4 = (a+(b-c))
    x = x1*x2*x3*x4
    if x < 0.:
        return 0.
    area = 0.25 * (x ** 0.5)
    return area


def RemoveZeroAreaTris(pts, triangles):
    filteredtris = []
    angleCache = {}

    for tri in triangles:
        area = HeronsFormula(pts, tri)

        ang1 = CalcTriangleAng(pts, angleCache, tri[2], tri[0], tri[1])
        ang2 = CalcTriangleAng(pts, angleCache, tri[0], tri[1], tri[2])
        ang3 = CalcTriangleAng(pts, angleCache, tri[1], tri[2], tri[0])

        if ang1 == 0. or ang2 == 0. or ang3 == 0.:
            continue

        if ang1 == math.pi or ang2 == math.pi or ang3 == math.pi:
            continue

        if area == 0.:
            continue

        filteredtris.append(tri)
    return filteredtris


def PySHull(pts):
    # S-hull: a fast sweep-hull routine for Delaunay triangulation by David Sinclair
    # http://www.s-hull.org/

    # Select seed point
    seedIndex = 0

    # Sort by radial distance
    radialSorted = RadialDistance(pts, seedIndex)

    # Nearest point to seed point
    nearestToSeed = radialSorted[1][1]

    # Find third point that creates the smallest circum-circle
    sortedCircumCircles = FindSmallestCircumCircle(pts, seedIndex, nearestToSeed)
    if sortedCircumCircles[0][0] == float("inf"):
        raise Exception("Invalid circumcircle error")
    thirdPtIndex = sortedCircumCircles[0][1]

    # Order points to be right handed
    crossProd = RightHandedCheck(pts, seedIndex, nearestToSeed, thirdPtIndex)
    if crossProd < 0.:
        # Swap points
        secondPtInd = thirdPtIndex
        thirdPtIndex = nearestToSeed
    else:
        # Already right handed
        secondPtInd = nearestToSeed

    # Centre of circum-circle
    centre = CircumCircleCentre(pts[seedIndex], pts[secondPtInd], pts[thirdPtIndex])

    # Sort points by distance from circum-circle centre
    dists = []
    for ptNum, pt in enumerate(pts):
        if ptNum == seedIndex:
            continue
        if ptNum == secondPtInd:
            continue
        if ptNum == thirdPtIndex:
            continue

        dist = CalcDist(pts[ptNum], centre)
        dists.append((dist, ptNum))
    dists.sort()
    orderToAddPts = [v[1] for v in dists]

    # Form triangles by sequentially adding points
    hull, triangles = FormTriangles(pts, (seedIndex, secondPtInd, thirdPtIndex), orderToAddPts)

    # Flip adjacent pairs of triangles to meet Delaunay condition
    # https://en.wikipedia.org/wiki/Delaunay_triangulation#Visual_Delaunay_definition:_Flipping
    delaunayTris = FlipTriangles(pts, triangles)

    # Remove zero area triangles
    filteredTris = RemoveZeroAreaTris(pts, delaunayTris)

    return filteredTris


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import sys
    import time
    import numpy as np

    n = 50
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])

    pts = np.random.rand(n, 2)

    startTime = time.time()
    triangles = PySHull(pts)
    print("Processed", n, "points in", time.time() - startTime, "sec")

    for tri in triangles:
        tri2 = list(tri[:])
        tri2.append(tri[0])
        plt.plot(pts[tri2, 0], pts[tri2, 1])

    # plt.plot(pts[:,0], pts[:,1], 'x')
    plt.show()
