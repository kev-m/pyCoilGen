import numpy as np


def smooth_track_by_folding(track_in, smoothing_length):
    """
    Smooth a track by folding its data.

    Args:
        track_in (ndarray): The input track data with shape (2, m).
        smoothing_length (int): The length of smoothing.

    Returns:
        ndarray: The smoothed track with shape (2, m).
    """
    track_out = track_in.copy()

    if smoothing_length > 1:
        extended_track = np.concatenate((np.tile(track_in[:, 0][:, np.newaxis], smoothing_length),
                                        track_in[:, 1:-1], np.tile(track_in[:, -1][:, np.newaxis], smoothing_length)), axis=1)

        for shift_ind in range(-(smoothing_length - 1), smoothing_length):
            add_track = np.roll(extended_track, shift_ind, axis=1)
            add_track = add_track[:, smoothing_length:(-smoothing_length + 1)]
            track_out += add_track

        track_out /= (2 * smoothing_length - 1)

    return track_out
