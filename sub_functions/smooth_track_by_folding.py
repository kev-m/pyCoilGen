import numpy as np

def smooth_track_by_folding(track_in, smoothing_length):
    """
    Smooth a track by folding its data.

    Args:
        track_in (ndarray): The input track data with shape (2, M).
        smoothing_length (int): The length of smoothing.

    Returns:
        ndarray: The smoothed track with shape (2, M).
    """

    if smoothing_length > 1:
        track_out = track_in.copy()
        num_points = track_in.shape[1]

        # Extend the track by repeating the first and last points for smoothing
        extended_track = np.hstack([np.tile(track_in[:, 0].reshape(-1, 1), (1, smoothing_length)), 
                                    track_in, 
                                    np.tile(track_in[:, -1].reshape(-1, 1), (1, smoothing_length))])

        for shift_ind in range(-(smoothing_length - 1), smoothing_length):
            # Shift the extended track to create overlapping segments
            add_track = np.roll(extended_track, shift_ind, axis=1)

            # Remove the overlapping parts to avoid double counting
            add_track = add_track[:, smoothing_length:(num_points + smoothing_length)]

            # Add the overlapping segments to the original track_out
            track_out += add_track

        # Calculate the average of the overlapping segments
        track_out /= (2 * smoothing_length)
        return track_out
    else:
        return track_in

