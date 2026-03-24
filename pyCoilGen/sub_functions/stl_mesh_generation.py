import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

radius = 0.07
nr = 30 # number of radial divisions
nt = 90 # 180 for 2 degree resolution, 90 for 4 degree resolution
inner_radius = radius / nr
z_positions = [-0.03675, 0.03675]
filename = "data/pyCoilGenData/Geometry_Data/Tenacity_circular_1.stl"


def create_mesh(radius,nr,nt,z):
    """
    Creates a mesh for a cylindrical surface.
    Generates vertices and faces for a mesh representing a cylindrical surface
    with a specified radius and height. The mesh is created by arranging vertices
    in concentric circles and connecting them with triangular faces.
    Args:
        radius (float): The outer radius of the cylindrical mesh.
        nr (int): The number of radial divisions (rings).
        nt (int): The number of theta (angular) divisions (sectors).
        z (float): The z-coordinate (height) of the mesh plane.
    Returns:
        tuple: A tuple containing:
            - verts (np.ndarray): An (nr*nt, 3) array of vertex coordinates [x, y, z].
            - faces (np.ndarray): An (n, 3) array of triangular face indices.
    Note:
        This function assumes `inner_radius` is defined in the outer scope.
        The mesh is created as a flat cylindrical annulus at height z.
    """


    verts=[]
    faces=[]

    for i in range(nr):

        r = inner_radius + (radius-inner_radius)*i/(nr-1)

        for j in range(nt):

            theta=2*np.pi*j/nt

            x=r*np.cos(theta)
            y=r*np.sin(theta)

            verts.append([x,y,z])

    verts=np.array(verts)

    def vid(i,j):
        return i*nt+j

    for i in range(nr-1):

        for j in range(nt):

            v1=vid(i,j)
            v2=vid(i,(j+1)%nt)
            v3=vid(i+1,(j+1)%nt)
            v4=vid(i+1,j)

            faces.append([v1,v2,v3])
            faces.append([v1,v3,v4])

    return verts,np.array(faces)


def write_stl(filename,verts,faces):
    """
    Write a mesh to an STL (Stereolithography) file format.
    This function takes a set of vertices and faces defining a 3D mesh and writes
    them to a binary STL file. It ensures that all triangle normals point outward
    by flipping triangles as needed based on the Z-component of the normal vector.
    Args:
        filename (str): The path and name of the output STL file to be created.
        verts (numpy.ndarray): An array of vertices with shape (N, 3) containing
            the 3D coordinates of each vertex in the mesh.
        faces (list or numpy.ndarray): A list or array of triangular faces where
            each face is defined by three indices referring to vertices in the
            verts array.
    Returns:
        None: The function writes directly to a file instead of returning a value.
    Notes:
        - The function modifies the faces array in place to ensure consistent
            triangle winding order.
        - Triangle normals with negative Z-components are flipped to ensure
            outward-facing normals.
        - Degenerate triangles (with zero area) have their normal set to [0, 0, 0].
        - The output file uses binary STL format with an 80-byte header and
            4-byte face count.
    Raises:
        IOError: If the file cannot be opened or written to.
        ValueError: If faces contain invalid vertex indices.
    """


    for i,tri in enumerate(faces):

        v1,v2,v3 = verts[tri]

        normal = np.cross(v2-v1, v3-v1)

        if normal[2] < 0:  # flip triangle
            faces[i] = [tri[0], tri[2], tri[1]]

    with open(filename,"wb") as f:

        f.write(b'PythonSTL'+b' '*(80-len('PythonSTL')))
        f.write(struct.pack("<I",len(faces)))

        for tri in faces:

            v1,v2,v3=verts[tri]

            normal=np.cross(v2-v1,v3-v1)
            norm=np.linalg.norm(normal)

            if norm==0:
                n=[0,0,0]
            else:
                n=normal/norm

            f.write(struct.pack("<3f",*n))

            for v in [v1,v2,v3]:
                f.write(struct.pack("<3f",*v))

            f.write(struct.pack("<H",0))


# ==========================================================
# CHECK MESH QUALITY
# ==========================================================
def check_mesh_quality(mesh, dsv_radius = 0.016, visualize = False):
    """
    Check and display the quality metrics of a mesh.
    This function validates mesh properties including bounds, centroid, vertex
    and face counts, and watertightness. Optionally visualizes the mesh with
    a reference DSV (Diameter of Spherical Volume) sphere.
    Args:
        mesh: A mesh object with properties like bounds, centroid, vertices,
            faces, and is_watertight.
        dsv_radius (float): The radius of the DSV sphere for reference in
            visualization. Defaults to 0.016.
        visualize (bool): If True, displays a 3D plot of the mesh vertices
            and the DSV sphere wireframe. Defaults to False.
    Returns:
        None
    """

    print("Mesh quality check:")
    print("Bounds:", mesh.bounds)
    print("Centroid:", mesh.centroid)
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Watertight:", mesh.is_watertight)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2], s=1)
        ax.set_title("STL Mesh Visualization")

        # Plot the DSV sphere for reference
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = dsv_radius * np.cos(u)
        y = dsv_radius * np.sin(u) * np.cos(v)
        z = dsv_radius * np.sin(u) * np.sin(v)
        ax.plot_wireframe(x, y, z, color='r', alpha=0.5, label='DSV Sphere')
        ax.legend()     
        plt.show()
        


def create_stl_mesh(radius,nr,nt,z_positions,filename):
    """
    Generate and save an STL mesh file from circular mesh layers.
    Creates a 3D mesh by stacking multiple circular mesh layers at different
    z-positions and exports the combined geometry to an STL file.
    Args:
        radius (float): The radius of the circular mesh layers.
        nr (int): Number of radial divisions in each mesh layer.
        nt (int): Number of tangential divisions in each mesh layer.
        z_positions (list or array-like): List of z-coordinates where mesh
            layers should be created.
        filename (str): Output file path for the generated STL file.
    Returns:
        None
    Raises:
        IOError: If the STL file cannot be written to the specified path.
    Example:
        >>> create_stl_mesh(radius=10, nr=20, nt=32,
        ...                 z_positions=[0, 5, 10], filename='mesh.stl')
        STL saved: mesh.stl
    """

    all_verts=[]
    all_faces=[]
    offset=0
    for z in z_positions:

        v,f=create_mesh(radius,nr,nt,z)

        all_verts.append(v)
        all_faces.append(f+offset)

        offset+=len(v)

    verts=np.vstack(all_verts)
    faces=np.vstack(all_faces)

    write_stl(filename,verts,faces)

    print("STL saved:",filename)