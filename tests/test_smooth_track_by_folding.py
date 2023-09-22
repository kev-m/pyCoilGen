import numpy as np

# Code under test
from pyCoilGen.sub_functions.smooth_track_by_folding import smooth_track_by_folding


def generate_triangular_waveform(cycles, magnitude, wavelength):
    """
    Generates a 2xM triangular waveform.

    Args:
        cycles (int): The number of repeats.
        magnitude (float): The magnitude of the waveform.
        wavelength (int): The wavelength of the waveform.

    Returns:
        numpy.ndarray: A 2xM array where index 0 represents X-values and index 1 represents Y-values.
    """

    length = cycles*wavelength

    # Generate X-values
    x_values = np.linspace(0, length, length)

    # Generate Y-values (triangular waveform)
    y_values = np.concatenate((np.linspace(0, magnitude, wavelength // 2),
                               np.linspace(magnitude, 0, wavelength // 2)))

    # Repeat the waveform to match the desired length
    y_values = np.tile(y_values, cycles)

    # Trim excess points if necessary
    x_values = x_values[:length]
    y_values = y_values[:length]

    return np.array([x_values, y_values])


def test_smooth_track_by_folding():
    input_data = generate_triangular_waveform(2, 1.0, 10)
    smoothing_length = 2
    ##########################################################
    # Function under test
    output_data = smooth_track_by_folding(input_data, smoothing_length=smoothing_length)
    ##########################################################

    assert input_data.shape == output_data.shape
    assert output_data[0, 0] == input_data[0, 1]/(2*smoothing_length-1)
    assert output_data[1, 0] == input_data[1, 1]/(2*smoothing_length-1)
    assert output_data[0, 1] == input_data[0, 1]
    assert output_data[1, 1] == input_data[1, 1]

test_smooth_track_by_folding()