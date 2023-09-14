import numpy as np

from typing import List

from .data_structures import TopoGroup, Mesh


def calc_mean_loop_normal(group: TopoGroup, coil_mesh: Mesh):
    """
    Calculate the mean loop normal for the given group.

    Args:
        group (Group): The group containing loops.
        coil_mesh (Mesh): The coil mesh object.

    Returns:
        ndarray: The mean loop normal.
    """

    # Initialize an array to store all the loop normals
    all_loop_normals = np.zeros((3, len(group.loops)))

    # Calculate loop normals for each loop in the group
    for loop_ind in range(len(group.loops)):
        group_center = np.mean(group.loops[loop_ind].v, axis=1)
        loop_vecs = group.loops[loop_ind].v[:, 1:] - group.loops[loop_ind].v[:, :-1]
        center_vecs = group.loops[loop_ind].v[:, :-1] - group_center[:, np.newaxis]

        # Calculate cross products to get loop normals
        loop_normals = np.cross(loop_vecs, center_vecs, axis=0)

        # Calculate mean loop normal
        all_loop_normals[:, loop_ind] = np.mean(loop_normals, axis=1)

    # Calculate the mean loop normal
    loop_normal = np.mean(all_loop_normals, axis=1)
    loop_normal /= np.linalg.norm(loop_normal)

    # Make sure the loop normal points outwards seen from the coordinate center
    if np.sum(loop_normal * group_center) < 0:
        loop_normal *= -1

    return loop_normal


"""
Please note that in the above code, we are assuming that Group is a data structure that holds the loop information and
Mesh is a data structure that holds the coil mesh information. Also, the loop_normal calculated in the Python code will
be an ndarray representing the mean loop normal.
"""
