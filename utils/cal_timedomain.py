import numpy as np
from scipy.stats import skew, kurtosis


def seg_with_thereshold(time_series, thershold):

    """
    inputs:
    time_series: 1d signal as a list or ndarray
    thershold: usually 90, 88 or 86

    return: n by m matrix, n as numbers of evnets.
    m[0]: event start idx
    m[-1]: event duration

    notes: works on non-temporal unit
    """

    below_segments = []
    start_index = None
    for i, value in enumerate(time_series):
        if value < thershold and start_index is None:
            start_index = i
        elif value >= thershold and start_index is not None:
            end_index = i - 1
            segment_length = end_index - start_index + 1
            below_segments.append([start_index, segment_length])
            start_index = None
    if start_index is not None:
        segment_length = len(time_series) - start_index
        below_segments.append([start_index, segment_length])

    return np.array(below_segments)

def culmulative_time(time_series, thershold, fs):

    """
    inputs:
    time_series: 1d signal as a list or ndarray
    thershold: usually 90, 88 or 86
    fs: sampling frequency

    return: the total time below the thershold
    notes: works on non-temporal unit
    """

    ### count all samples below the thershold
    below_thershold = np.sum(time_series < thershold)
    below_time = (below_thershold - 1) / fs

    if below_time < 0:
        return 0
    else:
        return below_time

class TimeDomain_tool_with_stages():

    """
    
    intput:
    x: 1d array, the signal of SpO2.
    stages: 1d array, the sleep stage of each 30s epoch.

    output:
    mean, min and max of the signal in each sleep stage.

    
    
    """
    def __init__(self, x, fs, stages = None):
        self.x = x
        self.stages = stages
        self.fs = int(fs)
        self.data = self.x.reshape(self.stages.size, 30*self.fs)
        # print(self.data)


    def parameters_with_phases(self, keys):
        list_phases = ['wake', 'light', 'deep', 'rem', 'all']
        if keys not in list_phases:
            raise KeyError('keys must be in {}'.format(list_phases))
        else:
            if keys == 'wake':
                data = self.data[self.stages == 0]
            elif keys == 'light':
                data_1 = self.data[self.stages == 1]
                data_2 = self.data[self.stages == 2]
                data = np.concatenate((data_1, data_2), axis=0)   
            elif keys == 'deep':
                data = self.data[self.stages == 3]
            elif keys == 'rem':
                data = self.data[self.stages == 5]
            else:
                data = self.data

        if data.shape[0] == 0:
            return np.mean(self.x), np.min(self.x), np.max(self.x)
        else:
            return np.mean(data), np.min(data), np.max(data)        

    def all_variance(self):
        return np.var(self.x)
    
    def all_skewness(self):
        return skew(self.x)

    def all_kurtosis(self):
        return kurtosis(self.x)
    
    def all_minimum(self):
        return np.nanmin(self.x)
    
    def all_cumulative_time(self, thershold, fs):
        
        # tmp = seg_with_thereshold(self.x, thershold=thershold)
        # ct = 0
        # for idx, [start, samples] in enumerate(tmp):
        #     if samples <= fs * 2 or samples >= fs * 120:
        #         pass
        #     else:
        #         ct = ct + samples

        # return ct/fs
        return culmulative_time(self.x, thershold, fs)
    

if __name__ == '__main__':
    def id_2_data_str(org_dir, id):
        return org_dir+str(id)+'.npy'
    def id_2_label_str(org_dir, id):
        return 'E:/datasets/SHHS1/label/raw_label/shhs1-' + str(id) + '-nsrr.xml'
    # from tool_read_xml import label_start_duration, stage_on_nsrr
    # org_dir = '../shhs_only_spo2/'
    # id_list = 205002
    # data_path = id_2_data_str(org_dir, id_list)
    # label_path = id_2_label_str(org_dir, id_list)
    # data_spo2 = np.load(data_path)
    # stages = stage_on_nsrr(label_path)
    # events = label_start_duration(label_path)

    # print(TimeDomain_tool_with_stages(x=data_spo2, stages=stages).parameters_with_phases('rem'))
    # print(TimeDomain_tool_with_stages(x=data_spo2, stages=stages).all_cumulative_time(thershold=86, fs=1))