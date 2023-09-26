import hashlib
from sys import getsizeof
import struct

# Logging
import logging

# Local imports
from .data_structures import CoilSolution

log = logging.getLogger(__name__)


def temp_evaluation(coil_solution: CoilSolution, input, target_field):
    """
    Evaluates whether pre-calculated values can be used from previous calculations.

    Args:
        CoilSolution
        input (object): Input parameters.
        target_field (object): Target field data.

    Returns:
        object: Updated input parameters.
    """
    preoptimization_input_hash = generate_DataHash([input.coil_mesh_file, input.iteration_num_mesh_refinement,
                                                    input.surface_is_cylinder_flag, target_field])
    optimized_input_hash = generate_DataHash([input.sf_opt_method, input.tikhonov_reg_factor, input.fmincon_parameter])

    coil_solution.optimisation.use_preoptimization_temp = False
    coil_solution.optimisation.use_optimized_temp = False

    # Initialize values if not existing
    if not hasattr(coil_solution.optimisation, 'preoptimization_hash'):
        coil_solution.optimisation.preoptimization_hash = 'none'

    if not hasattr(coil_solution.optimisation, 'optimized_hash'):
        coil_solution.optimisation.optimized_hash = 'none'

    if preoptimization_input_hash == coil_solution.optimisation.preoptimization_hash:
        coil_solution.optimisation.use_preoptimization_temp = True

    if optimized_input_hash == coil_solution.optimisation.optimized_hash:
        coil_solution.optimisation.use_optimized_temp = True

    # Assign the new hash to temp
    coil_solution.optimisation.preoptimization_hash = preoptimization_input_hash
    coil_solution.optimisation.optimized_hash = optimized_input_hash

    log.debug(" - preoptimization_hash: %s", preoptimization_input_hash)
    log.debug(" - optimized_hash: %s", optimized_input_hash)

    return input


def generate_DataHash(Data):
    """
    Generates the MD5 hash of the provided data.

    Args:
        Data: Data to be hashed.

    Returns:
        str: MD5 hash of the data.
    """
    Engine = hashlib.md5()
    H = CoreHash(Data, Engine)
    H = H.hexdigest()  # To hex string
    return H


def CoreHash(Data, Engine):
    """
    Core hashing function for generating MD5 hash recursively.

    Args:
        Data: Data to be hashed.
        Engine: MD5 engine.

    Returns:
        hashlib.md5: MD5 hash object.
    """
    # Consider the type of empty arrays
    S = f"{type(Data).__name__} {Data.shape}" if hasattr(Data, 'shape') else f"{type(Data).__name__}"
    Engine.update(S.encode('utf-8'))
    H = hashlib.md5(Engine.digest())

    if isinstance(Data, dict):
        for key in sorted(Data.keys()):
            H.update(CoreHash(Data[key], hashlib.md5()).digest())

    elif isinstance(Data, (list, tuple)):
        for item in Data:
            H.update(CoreHash(item, hashlib.md5()).digest())

    elif isinstance(Data, (str, bytes)):
        H.update(Data.encode('utf-8'))

    elif isinstance(Data, bool):
        H.update(int(Data).to_bytes(1, 'big'))

    elif isinstance(Data, int):
        H.update(int(Data).to_bytes(getsizeof(int), 'big'))

    elif isinstance(Data, float):
        H.update(bytearray(struct.pack("f", float(Data))))

    elif callable(Data):
        H.update(CoreHash(Data.__code__, hashlib.md5()).digest())

    else:
        log.warning(f"Type of variable not considered: {type(Data).__name__}")

    return H
