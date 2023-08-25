# System imports
import numpy as np
import os

# Logging
import logging

from typing import List

# Local imports
from sub_functions.data_structures import CoilPart, CoilSolution

log = logging.getLogger(__name__)

def calculate_inductance_by_coil_layout(solution: CoilSolution, input_args) -> CoilSolution:
    """
    Calculate inductance by means of integration of the vector potential along the coil track.

    Args:
        solution (CoilSolution): List of CoilPart structures.
        input_args: Input arguments.

    Returns:
        CoilSolution: A data structure containing inductance and other related values.
    """
    
    conductor_width = input_args.conductor_cross_section_width
    conductor_height = input_args.conductor_cross_section_height
    skip_inductance_calculation = input_args.skip_inductance_calculation

    down_sample_factor = 10

    # Initialize CoilSolution attributes for each coil part
    coil_parts = solution.coil_parts
    for coil_part in coil_parts:
        coil_part.coil_resistance = 0
        coil_part.coil_inductance = 0
        coil_part.coil_length = 0
        coil_part.coil_cross_section = 0

    if not skip_inductance_calculation:
        sim_freq = 1
        material_conductivity = 5.8e7

        # Check if FastHenry2 is installed
        try:
            result = os.system('WHERE /F /R "c:\Program Files (x86)" FastHenry2.exe')
        except:
            result = 1

        if result == 0:
            for part_ind in range(len(coil_parts)):
                script_file = create_fast_henry_file(coil_part.wire_path, conductor_width, conductor_height, sim_freq, material_conductivity, down_sample_factor)
                coil_part.coil_resistance, coil_part.coil_inductance, coil_part.coil_cross_section = create_fast_henry_file(coil_part.wire_path, conductor_width, conductor_height, sim_freq, material_conductivity, down_sample_factor)
        else:
            print(' FastHenry2 is not installed in the Folder- "Program Files (x86)" -", magnetic inductance will not be calculated.. ')
            for coil_part in coil_parts:
                coil_part.coil_resistance = 0
                coil_part.coil_inductance = 0
                coil_part.coil_length = 0
                coil_part.coil_cross_section = 0
    else:
        for coil_part in coil_parts:
            coil_part.coil_resistance = 0
            coil_part.coil_inductance = 0
            coil_part.coil_length = 0
            coil_part.coil_cross_section = 0

    # Calculate the length of the coil
    for part_ind in range(len(coil_parts)):
        wire_path = coil_part.wire_path
        coil_part.coil_length = np.sum(np.linalg.norm(wire_path.v[:, 1:] - wire_path.v[:, :-1], axis=0))

    # Return the modified coil_parts list with calculated attributes
    return coil_parts


def create_fast_henry_file(wire_path, conductor_width, conductor_height, sim_freq, material_conductivity, down_sample_factor):
    """
    Create a FASTHENRY2 input file, run it with fasthenry2 and read out the result.

    Args:
        wire_path: Sequence of 3D points coords in m. A dictionary with 'uv' and 'v' keys.
        conductor_width: Width of conductor in m.
        conductor_height: Height of conductor in m.
        sim_freq: Frequency in Hz.
        material_conductivity: Conductivity in 1/(mm*ohm).
        down_sample_factor: Down sampling factor.

    Returns:
        tuple: coil_resistance, coil_inductance, coil_cross_section
    """

    # Extract downsampled wire_path data
    wire_path_downsampled_v = wire_path['v'][:, ::down_sample_factor]

    # Create the ".inp" input file
    fast_henry_file_name = 'coil_track_FH2_input.inp'
    with open(fast_henry_file_name, 'w') as fid:
        fid.write('\n\n\n.Units m\n')  # Specify the unit system
        fid.write(f".Default sigma={material_conductivity}\n\n")
        
        # Define the points (vertices) in the FASTHENRY input file
        for point_ind in range(wire_path_downsampled_v.shape[1]):
            str_to_include = (
                f"\nN{point_ind+1} "
                f"x={wire_path_downsampled_v[0, point_ind]} "
                f"y={wire_path_downsampled_v[1, point_ind]} "
                f"z={wire_path_downsampled_v[2, point_ind]}"
            )
            fid.write(str_to_include)
        
        fid.write('\n\n\n')
        
        # Define the cross section of the wire segment
        for seg_ind in range(1, wire_path_downsampled_v.shape[1]):
            str_to_include = (
                f"\nE{seg_ind} "
                f"N{seg_ind} "
                f"N{seg_ind+1} "
                f"w={conductor_width} "
                f"h={conductor_height}"
            )
            fid.write(str_to_include)
        
        fid.write(f"\n\n.external N1 N{wire_path_downsampled_v.shape[1]}")
        fid.write(f"\n\n.freq fmin={sim_freq} fmax={sim_freq} ndec=1")
        fid.write("\n\n.end")

        return fast_henry_file_name

def execute_fast_henry_file_script_windows(fast_henry_file_name: str, conductor_width: float, conductor_height: float, sim_freq: float):
    # Create the Windows ".vbs" script file to run the ".inp" automatically
    script_file_name = 'run_FH2.vbs'
    with open(script_file_name, 'w') as fid2:
        fid2.write('Set FastHenry2 = CreateObject("FastHenry2.Document")\n')
        fid2.write(f'couldRun = FastHenry2.Run("{os.getcwd()}\\{fast_henry_file_name}")\n')
        fid2.write('Do While FastHenry2.IsRunning = True\n')
        fid2.write('  Wscript.Sleep 500\n')
        fid2.write('Loop\n')
        fid2.write('inductance = FastHenry2.GetInductance()\n')
        fid2.write('FastHenry2.Quit\n')
        fid2.write('Set FastHenry2 = Nothing\n')

    # Run the script
    os.system(f'wscript {script_file_name}')

    # Read the results
    with open('Zc.mat') as fid3:
        out = fid3.readlines()
    real_Z = float(out[2])
    im_Z = float(out[3])

    coil_cross_section = conductor_width * conductor_height  # in m²
    coil_inductance = im_Z / (sim_freq * 2 * np.pi)  # in Henry
    coil_resistance = real_Z  # in Ohm

    # Remove the created script files
    os.remove('Zc.mat')
    os.remove('run_FH2.vbs')
    os.remove('coil_track_FH2_input.inp')

    return coil_resistance, coil_inductance, coil_cross_section
