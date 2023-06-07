# System imports
from typing import List
import numpy as np
# Logging
import logging

# Local imports
from sub_functions.data_structures import DataStructure, Mesh

log = logging.getLogger(__name__)


def split_disconnected_mesh(coil_mesh_in : Mesh) -> List[DataStructure]:
    """
    Split the mesh and the stream function if there are disconnected pieces, such as shielded gradients.

    Args:
        coil_mesh_in (Mesh): Input coil mesh object with attributes 'faces' and 'vertices'.

    Returns:
        coil_parts (list): List of coil parts, each containing a separate Mesh.

    """

    return coil_mesh_in.separate_into_get_parts()
