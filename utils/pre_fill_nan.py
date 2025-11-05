import numpy as np


def fill_nan_with_threshold(sequence, threshold):

    """
    In the global method, this is step 1.

    sequence: list or np.array.

    threshold: int, usually 25 in our project.

    Replace the values in the sequence that are less than or equal to the threshold.
    Then interpolate the values that are nan in the sequence based on the values that are not nan.
    """


    sequence = np.array(sequence).astype(float)
    sequence[sequence <= threshold] = np.nan
    mask = np.isnan(sequence)
    sequence[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), sequence[~mask])
    return np.round(sequence).astype(int)
