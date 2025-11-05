import numpy as np
from scipy.signal import find_peaks
import itertools


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


def algorithms_combined(signal_spo2: np.ndarray, fs: int, bottoms: np.ndarray, arm_len = [0.5, 0.5]) -> np.ndarray:

    win_len_left = int(arm_len[0] * fs * 60) + 1  # left window length, default as 2mins
    win_len_right = int(arm_len[1] * fs * 60) + 1  # right window length 2mins, default as 2mins
    res = []

    for i in range(0, bottoms.shape[0]):
        val = bottoms[i]
        x = signal_spo2[val]
        pending_signal_l = signal_spo2[val - win_len_left: val]
        pending_idx_l = np.where(pending_signal_l >= x * (1 / 0.97))[0]
        pending_idx_r = np.where(signal_spo2[val: val + win_len_right] >= x * (1.01))[0]

        if len(pending_idx_l) and len(pending_idx_r) > 0:
                
            t0 = np.where(signal_spo2[val - win_len_left: val] == max(signal_spo2[val - win_len_left: val]))[0][-1]
            t1 = np.where(signal_spo2[val: val + win_len_right] == max(signal_spo2[val: val + win_len_right]))[0][0]
            
            res.append([val - win_len_left + t0, val, val + t1])
    return np.array(res)

def find_duplicate_trinity(mat: np.ndarray, col) -> np.ndarray:
    all_indices = np.arange(mat.shape[0])
    unique_block_nadirs, unique_block_nadirs_idx = np.unique(mat[:, col], return_index=True)
    duplicate_indices = []
    for value in unique_block_nadirs:
        row_indices = np.where(mat[:, col] == value)[0]
        if len(row_indices) > 1:
            duplicate_indices.append(row_indices)
    
    dupliate_list = list(itertools.chain(*duplicate_indices))

    unique_idx_manually = []
    for x in all_indices:
        if x not in dupliate_list:
            unique_idx_manually.append(x)
    

    # ## in unique_block_nadirs_idx, if any element is in duplicate_indices, then remove it.
    # unique_block_nadirs_idx = [val for idx, val in enumerate(unique_block_nadirs_idx) if idx not in np.array(duplicate_indices)]

    return np.array(unique_idx_manually), duplicate_indices


def find_range_max(dupliate_idx, block_trinity):
    res = []
    for i, idx_group in enumerate(dupliate_idx):
        repeat_nadirs_blocks = block_trinity[idx_group]
        ## range = forward - backward
        group_range = repeat_nadirs_blocks[:, 2] - repeat_nadirs_blocks[:, 0]
        max_idx = np.argmax(group_range)
        res.append(idx_group[max_idx])
        
    return np.array(res)


def find_intersection(trinity_blocks):
    drop = []
    left, right = trinity_blocks[0, 0], trinity_blocks[0, 2]
    for i, block in enumerate(trinity_blocks):
        if i == 0:
            continue
        if block[0] < right:
            drop.append(i)
        else:
            right = block[2]
    
    return np.array(drop)

def remove_duplicate_trinity(mat: np.ndarray) -> np.ndarray:

    # step 1: find the duplicate start
    unique_backwards_idx, dupliate_backwards_idx = find_duplicate_trinity(mat, 0)
    dupliate_backwards_idx = find_range_max(dupliate_backwards_idx, mat)
    usefull_backward_idx = np.concatenate((unique_backwards_idx, dupliate_backwards_idx))
    unique_backwards_idx = np.sort(usefull_backward_idx).astype(int)
    mat = mat[unique_backwards_idx]

    # step 2: find the duplicate end
    unique_forwards_idx, dupliate_forwards_idx = find_duplicate_trinity(mat, 2)
    dupliate_forwards_idx = find_range_max(dupliate_forwards_idx, mat)
    usefull_forward_idx = np.concatenate((unique_forwards_idx, dupliate_forwards_idx))
    usefull_forward_idx = np.sort(usefull_forward_idx).astype(int)
    mat = mat[usefull_forward_idx]
    ##  sort the block_trinity by the backward index.
    mat = mat[np.argsort(mat[:, 0])]

    # step 3: find the intersection

    drop_idx = find_intersection(mat)
    if len(drop_idx) > 0:
        mat = np.delete(mat, drop_idx, axis=0)
    # print(drop_idx)
    return mat

def revise_nadir(trinity_blocks, signal_y):
    res = []
    for block in trinity_blocks:
        backward, nadir, forward = block
        nadir_new = np.where(signal_y[backward: forward] == min(signal_y[backward: forward]))[0][0] + backward
        res.append([backward, nadir_new, forward])
    return np.array(res)


def events_main(data, fs):
    y = fill_nan_with_threshold(data, 25)
    peaks, _ = find_peaks(-y)
    block_trinity = algorithms_combined(y, fs, peaks, arm_len=[0.5, 0.5])
    block_trinity = remove_duplicate_trinity(block_trinity)
    block_trinity = revise_nadir(block_trinity, y)
    return block_trinity