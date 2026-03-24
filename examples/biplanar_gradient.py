import sys
sys.path.append('.')
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import logging
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.pyCoilGen_planar import pyCoilGen_planar
from scipy.special import sph_harm_y
# from pyCoilGen.sub_functions.utils import *
from pyCoilGen.sub_functions.stl_mesh_generation import *
from pyCoilGen.sub_functions.simulate_gradient_coil import simulate_gradient_coil
from pyCoilGen.sub_functions.gradient_former import generate_gradient_former
from pyCoilGen.helpers.metrics import print_metrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ==========================================================
# MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    """Planar MRI gradient coil design pipeline using pyCoilGen.
    This script demonstrates a complete end-to-end workflow for designing,
    simulating, evaluating, and exporting a biplanar MRI gradient coil.
    Workflow:
        1. Create and save an STL mesh for the current-carrying planar surfaces.
        2. Validate mesh quality and visualize geometry.
        3. Configure pyCoilGen optimization and discretization parameters.
        4. Run planar coil synthesis with ``pyCoilGen_planar``.
        5. Simulate magnetic field behavior over the DSV with Magpylib helpers.
        6. Compute and print gradient performance metrics.
        7. Export coil former and wire-channel STL files for fabrication.
    Attributes:
        PLATE_SIZE (float): Plate width/height in meters (6 in).
        PLATE_TOP (float): Z position of the top plate in meters.
        PLATE_BOTTOM (float): Z position of the bottom plate in meters.
        DSV_IMAGING (float): Diameter of the spherical imaging region in meters.
        CNC (dict[str, float]): Fabrication and conductor parameters:
            - ``diameter``: Conductor diameter in meters.
            - ``current``: Current per contour in amperes.
            - ``thickness``: Wire thickness / cut width in meters.
            - ``width``: Wire width / normal shift length in meters.
        MESH_RESOLUTION (int): General mesh-resolution control value.
    Notes:
        - The target field is configured through ``field_shape_function`` (e.g., "z").
        - ``target_gradient_strength`` is interpreted in T/m at 1 A drive current.
        - Output artifacts (plots, debug data, exported STL files) are written to
        the configured project/output directories.
    ==============================================================
    PLANAR MRI GRADIENT COIL DESIGN PIPELINE
    ==============================================================
    This example demonstrates the design of a biplanar gradient coil using pyCoilGen. The pipeline includes the following steps:
    1. Design the STL file for the current-carrying surface and save it.
    2. Check the STL file quality and visualize the geometry.
    3. Configure pyCoilGen inputs for the coil design.
    4. Run pyCoilGen to obtain the coil design and wire patterns.
    5. Simulate the coil design using Magpylib to obtain the magnetic field distribution.
    6. Evaluate the performance metrics of the gradient coil design.
    7. Write the STL files for the coil parts (plates and wire channels) for CNC machining or 3D printing.
    """

    log.info("\n========================================")
    log.info("PLANAR GRADIENT COIL DESIGN PIPELINE")
    log.info("========================================\n")

    # ==========================================================
    # GLOBAL PARAMETERS
    # ==========================================================

    PLATE_SIZE = 152.4e-3 # in meters, 6 inches
    PLATE_TOP = 0.03675
    PLATE_BOTTOM = -0.03675
    DSV_IMAGING = 0.032 # in meters, diameter of the spherical imaging region
    CNC = {
        "diameter": 2e-3,   # meters
        "current": 1.0,         # current per contour (amps)
        "thickness": 1.6e-3, # meters, thickness of the wire (overloaded as interconnection cut width)
        "width": 2e-3,       # meters, width of the wire (overloaded as normal shift length)
    }
    MESH_RESOLUTION = 40

    # ==========================================================
    # STEP 1 - DESIGN THE STL FILE FOR CURRENT CARRYING SURFACE - SAVE STL
    # ==========================================================
    stl_path = "data/pyCoilGenData/Geometry_Data/Tenacity_circular_1.stl"
    radius = 0.07
    nr = 30 # number of radial divisions
    nt = 90 # 180 for 2 degree resolution, 90 for 4 degree resolution
    inner_radius = radius / nr
    z_positions = [-0.03675, 0.03675]
    create_stl_mesh(radius,nr,nt,z_positions,stl_path)

    #=========================================================
    # STEP 2 - CHECK THE STL FILE QUALITY AND VISUALIZE THE GEOMETRY
    #========================================================
    mesh = trimesh.load(stl_path)
    check_mesh_quality(mesh, dsv_radius=DSV_IMAGING * 0.5)
    
    #=========================================================
    # STEP 3 - CONFIGURE PYCOILGEN INPUTS
    #=========================================================
    arg_dict = {
        'field_shape_function': 'z',  # definition of the target field
        'coil_mesh_file': stl_path.split('/')[-1],# assumes the STL file is in the Geometry_Data folder 
        # 'target_mesh_file': None,
        'secondary_target_mesh_file': 'none',
        # 'secondary_target_weight': 0.5,
        'target_region_radius': DSV_IMAGING * 0.5,  # in meter
        # 'target_region_resolution': 10,  # MATLAB 10 is the default
        'target_region_resolution': 10,  # number of points along each axis in the target region; higher values lead to better optimization results but longer runtime
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'none',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 20, # was 12
        # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'pot_offset_factor': 0.35,
        'surface_is_cylinder_flag': False,
        # the width for the interconnections are interconnected; in meter
        'interconnection_cut_width': CNC['thickness'], # overloaded as wire_depth 
        # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'normal_shift_length': CNC['width'], # overloaded as wire_spacing = 2mm for AWG14 and 0.3mm for CNC
        'iteration_num_mesh_refinement': 1,  # the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],
        # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'tikhonov_reg_factor': 200,  # Tikhonov regularization factor for the SF optimization
        'output_directory': 'images',  # [Current directory]
        'project_name': 'biplanar_xgradient_Tenacity',
        'persistence_dir': 'debug',
        'debug': DEBUG_VERBOSE,
        'target_field_weighting': True,
        'minimize_method_parameters': "{'tol': 1e-9}",
        'minimize_method_options': "{'disp': True, 'maxiter' : 5}",
        'target_gradient_strength': 1, # in T/m, this is the target gradient strength at the center of the DSV; the optimization will try to achieve this strength with 1 A of current
        'skip_postprocessing': True,
        'level_set_method': 'primary',
        'sf_opt_method': 'Iterative',
        # 'sf_dest_file': 'test_grad_design_preopt',
        # 'sf_source_file': 'test_grad_design_preopt',
        # 'sf_source_file': 'none',
    }

    # ==========================================================
    # STEP 4 - RUN PYCOILGEN TO OBTAIN THE COIL DESIGN AND WIRE PATTERNS
    # ==========================================================
    log.info("Running pyCoilGen planar...")
    coil_parts = pyCoilGen_planar(log, arg_dict)
    log.info("pyCoilGen finished")

    #=========================================================
    # STEP 5 - SIMULATE THE COIL DESIGN USING MAGPYLIB TO OBTAIN THE MAGNETIC FIELD DISTRIBUTION
    #=========================================================
    # simulation contains - B, Bx, By, Bz, points (where the field is evaluated), and the coil geometry
    simulation = simulate_gradient_coil(coil_parts, DSV_IMAGING, wire=CNC, display_field=True)

    # ========================================================
    # STEP 6 - EVALUATE THE PERFORMANCE METRICS OF THE GRADIENT COIL DESIGN
    # ========================================================
    log.info("Evaluating performance metrics...")
    print_metrics(simulation['Bz'], coords = simulation['points'], axis = arg_dict['field_shape_function']) 
    
    #==========================================================
    # STEP 7 - WRITE THE STL FILES FOR THE COIL PARTS (PLATES AND WIRE CHANNELS) FOR CNC MACHINING
    #==========================================================
    output_prefix = f"./images/{arg_dict['project_name']}_{arg_dict['field_shape_function']}"
    generate_gradient_former(
    coil_parts,
    output_prefix=output_prefix)


    log.info("\n========================================")
    log.info("PLANAR GRADIENT COIL DESIGN PIPELINE COMPLETED")
    log.info("========================================\n")