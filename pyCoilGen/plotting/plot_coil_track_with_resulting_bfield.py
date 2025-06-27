import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from typing import List, cast
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilSolution
from pyCoilGen.helpers.common import title_to_filename

log = logging.getLogger(__name__)

# Note: This function has been auto-converted from MATLAB to Python.
# Conversion by: Claude Sonnet 4, in Agent Mode, 27/6/2025.
# It has not been tested in any way.

def plot_coil_track_with_resulting_bfield(coil_layouts: List[CoilSolution], single_ind_to_plot: int, plot_title: str, save_dir=None, dpi=100):
    """
    Plot the final wire track together with the mesh and the layout field.
    
    Args:
        coil_layouts (List[CoilSolution]): List of CoilSolution objects.
        single_ind_to_plot (int): Index of the solution to plot.
        plot_title (str): Title of the plot.
        save_dir (str, optional): Directory to save the plot. If None, the plot is only displayed.
        dpi (int, optional): Resolution of the saved plot.
    
    Returns:
        None
    """
    # Extract layout current for 1A
    coil_solution = coil_layouts[single_ind_to_plot]
    layout_c_1A = coil_solution.solution_errors.combined_field_layout[2, :]
    
    # Plot the final wire track together with the mesh and the layout field
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(plot_title, fontsize=16)
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
    
    # Plot each coil part
    for part_ind in range(len(coil_layouts[single_ind_to_plot].coil_parts)):
        coil_part = coil_layouts[single_ind_to_plot].coil_parts[part_ind]
        
        # Plot the mesh with transparency
        vertices = coil_part.coil_mesh.v
        faces = coil_part.coil_mesh.f
        face_vertices = vertices[faces]
        
        # Create mesh surface with transparency
        poly = Poly3DCollection(face_vertices, facecolors='black', alpha=0.05, edgecolors='black', linewidths=0.1)
        ax.add_collection3d(poly)
        
        # Plot wire path if available, otherwise plot contour lines
        if hasattr(coil_part, 'wire_path') and coil_part.wire_path is not None:
            wire_path_v = coil_part.wire_path.v
            ax.plot(wire_path_v[0, :], wire_path_v[1, :], wire_path_v[2, :], 
                   linewidth=2, color=[0, 0.4470, 0.7410])
        else:
            # Plot contour lines if wire path is not available
            if hasattr(coil_part, 'contour_lines') and coil_part.contour_lines is not None:
                for loop in coil_part.contour_lines:
                    if hasattr(loop, 'v'):
                        ax.plot(loop.v[0, :], loop.v[1, :], loop.v[2, :], 
                               linewidth=2, color=[0, 0.4470, 0.7410])
    
    # Set axis labels
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    
    # Calculate field values for scatter plot (convert to mT)
    plot_vals_scatter = layout_c_1A * 1000
    
    # Create scatter plot of field values
    target_coords = coil_solution.target_field.coords
    scatter = ax.scatter(target_coords[0, :], target_coords[1, :], target_coords[2, :], 
                        c=plot_vals_scatter, cmap='viridis')
    
    # Set color bar limits and add color bar
    scatter.set_clim(vmin=np.min(plot_vals_scatter), vmax=np.max(plot_vals_scatter))
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Layout Bz [mT/A]')
    
    # Set title
    ax.set_title(f"{plot_title}: Layout Bz [mT/A]")
    
    # Set equal aspect ratio and view
    try:
        combined_mesh = coil_solution.combined_mesh.vertices
        min_values = np.min(combined_mesh, axis=0)
        max_values = np.max(combined_mesh, axis=0)
        
        ax.set_xlim(min_values[0], max_values[0])
        ax.set_ylim(min_values[1], max_values[1])
        ax.set_zlim(min_values[2], max_values[2])
    except AttributeError:
        # If vertices attribute doesn't exist, try to get bounds from coil parts
        all_vertices = []
        for coil_part in coil_solution.coil_parts:
            if hasattr(coil_part.coil_mesh, 'v') and coil_part.coil_mesh.v is not None:
                all_vertices.append(coil_part.coil_mesh.v)
        if all_vertices:
            combined_vertices = np.vstack(all_vertices)
            min_values = np.min(combined_vertices, axis=0)
            max_values = np.max(combined_vertices, axis=0)
            
            ax.set_xlim(min_values[0], max_values[0])
            ax.set_ylim(min_values[1], max_values[1])
            ax.set_zlim(min_values[2], max_values[2])
    
    # Set viewing angle (equivalent to view(45,45) in MATLAB)
    ax.view_init(elev=45, azim=45)
    
    # Set plot styling
    fig.patch.set_facecolor('white')
    ax.tick_params(labelsize=10, weight='bold')
    
    plt.tight_layout()
    
    # Save the plot if save_dir is provided
    if save_dir is not None:
        fname = f'{save_dir}/plot_coil_track_with_resulting_bfield_{title_to_filename(plot_title)}.png'
        plt.savefig(fname, dpi=dpi)
        log.info(f'Plot saved to {fname}')
    
    # Display the plot
    plt.show()