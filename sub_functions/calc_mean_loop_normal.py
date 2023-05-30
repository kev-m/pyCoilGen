import numpy as np

def calc_mean_loop_normal(group, coil_mesh):
    """
    Calculates the mean loop normal for a given group of loops in the coil mesh.

    Args:
        group: The group of loops.
        coil_mesh: The coil mesh.

    Returns:
        The mean loop normal.
    """
    all_loop_normals = np.zeros((3, len(group.loops)))

    for loop_ind in range(len(group.loops)):
        group_center = np.mean(group.loops[loop_ind].v, axis=1)
        loop_vecs = group.loops[loop_ind].v[:, 1:] - group.loops[loop_ind].v[:, :-1]
        center_vecs = group.loops[loop_ind].v[:, :-1] - group_center[:, None]
        all_loop_normals[:, loop_ind] = np.mean(np.cross(loop_vecs, center_vecs, axis=0), axis=1)

    loop_normal = np.mean(all_loop_normals, axis=1)
    loop_normal /= np.linalg.norm(loop_normal)

    group_center = np.mean(group_center)

    if np.dot(loop_normal, group_center) < 0:
        loop_normal *= -1

    return loop_normal
