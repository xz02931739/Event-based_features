import numpy as np
import xml.etree.ElementTree as ET
from scipy.integrate import trapezoid


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


def slope_draw(f0, f1, sig):

    """
    based on two points, draw the slope of the signal.

    input:
    f0: int, the start point of the slope.
    f1: int, the end point of the slope.
    sig: 1d array, the signal of SpO2.

    output:
    slope_idx: 1d array, the index of the slope.
    slope: 1d array, the value of the slope.

    This function is used to draw the slope of the SpO2 signal from f0 to f1.

    """

    line_slope_x = [f0, f1]
    line_slope_y = [sig[f0], sig[f1]]
    slope_idx = np.arange(f0, f1 + 1)
    slope = np.interp(slope_idx, line_slope_x, line_slope_y) 

    return slope_idx, slope


def parameters_calculation(signals_spo2_i, r_LEFT_drop, r_local_min, r_RIGHT_up, sampling_rate):

    """

    this function is used to calculate the parameters of the event.


    input:
    signals_spo2_i: 1d array, the signal of SpO2.
    r_LEFT_drop: int, the index of the left drop point.
    r_local_min: int, the index of the local minimum point.
    r_RIGHT_up: int, the index of the right up point.
    sampling_rate: int, the sampling rate of the signals.

    output:
    duration_0: float, the duration of the event.
    h1: float, the height of the down side.
    k1: float, the slope of the down side.
    h2: float, the height of the up side.
    k2: float, the slope of the up side.
    k3: float, the slope of the whole event.
    area: float, the area of the event.
    
    this function is used to calculate the parameters of the event.
    return the duration, the height, the slope and the area of the event.
    """

    slope_idx, slope = slope_draw(sig=signals_spo2_i, f0=r_LEFT_drop, f1=r_RIGHT_up)
    duration_0 = (r_RIGHT_up - r_LEFT_drop) / sampling_rate
    h1 = signals_spo2_i[r_LEFT_drop] - signals_spo2_i[r_local_min]
    k1 = h1 / ((r_RIGHT_up - r_local_min) / sampling_rate)
    h2 = signals_spo2_i[r_RIGHT_up] - signals_spo2_i[r_local_min]
    k2 = h2 / ((r_local_min - r_LEFT_drop) / sampling_rate)

    h3 = signals_spo2_i[r_RIGHT_up] - signals_spo2_i[r_LEFT_drop]
    k3 = h3 / ((r_RIGHT_up - r_LEFT_drop) / sampling_rate)
    area = trapezoid(y=slope[0: -1] - signals_spo2_i[r_LEFT_drop: r_RIGHT_up], dx=(1/sampling_rate))

    return duration_0, h1, k1, h2, k2, k3, area


