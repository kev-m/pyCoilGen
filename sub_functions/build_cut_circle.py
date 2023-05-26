import numpy as np

def build_cut_circle(center_point, cut_width):
    """
    Build a rectangular cut shape in the form of a circular opening.
    
    Parameters:
    - center_point (ndarray): Array containing the x and y coordinates of the center point.
    - cut_width (float): Width of the cut.
    
    Returns:
    - cut_circle (ndarray): Array containing the x and y coordinates of the circular cut shape.
    """
    
    circular_resolution = 10
    
    # Build circular cut shapes
    opening_angles = np.linspace(0, 2*np.pi, circular_resolution)
    opening_circle = np.column_stack((np.sin(opening_angles), np.cos(opening_angles)))

    # Create a circular opening cut
    cut_circle = opening_circle * (cut_width / 2) + np.tile(center_point, (circular_resolution, 1))  

    return cut_circle



if __name__ == "__main__":
    # Example usage
    center_point = np.array([0, 0])
    cut_width = 1.0
    points = build_cut_circle(center_point, cut_width)
    print(points)
""" 
[[ 0.00000000e+00  5.00000000e-01]
 [ 3.21393805e-01  3.83022222e-01]
 [ 4.92403877e-01  8.68240888e-02]
 [ 4.33012702e-01 -2.50000000e-01]
 [ 1.71010072e-01 -4.69846310e-01]
 [-1.71010072e-01 -4.69846310e-01]
 [-4.33012702e-01 -2.50000000e-01]
 [-4.92403877e-01  8.68240888e-02]
 [-3.21393805e-01  3.83022222e-01]
 [-1.22464680e-16  5.00000000e-01]]
"""

"""
Convert the MatLab code below to Python, with DocString comments:
function cut_circle=build_cut_circle(center_point,cut_witdh)
%build a rectangular cut shape

circular_resolution=10;

%build circular cut shapes
opening_circle=[sin(0:(2*pi)/(circular_resolution-1):2*pi); cos(0:(2*pi)/(circular_resolution-1):2*pi)];

% %create a circular opening cut
cut_circle=opening_circle.*repmat(cut_witdh/2,[2 1])+center_point;


end

"""