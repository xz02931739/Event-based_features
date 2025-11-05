import numpy as np
import pandas as pd
import os, glob
from scipy.integrate import trapezoid

from tqdm import tqdm

from utils.event_instances import events_main
from utils.balanced_clinical_tools import fill_nan_with_threshold, slope_draw


def assign_events_to_sleep_stages(events, sleep_stages, sampling_rate=1, epoch_duration=30):
    
    """
    first, create a series of time stamps for sleep stages based on sampling rate and epoch duration
    """

    ## step 1: create time stamps for sleep stages
    location_stamps = sleep_stages.repeat(epoch_duration * sampling_rate)

    ## step 2: for each event count the maximum sleep stage in the event duration
    assigned_stages = []
    for event in events:
        start_idx, nadir_idx, end_idx = event
        event_stages_counters = location_stamps[start_idx: end_idx]
        most_common_stage = pd.Series(event_stages_counters).mode()[0]
        
        assigned_stages.append(most_common_stage)

    return assigned_stages

            
class EventFeaturesStages():

    def __init__(self, data, blocks, fs, stages):
        self.data = data
        self.fs = fs
        self.blocks = blocks
        self.stages = stages
        
        self.assigned_stages = assign_events_to_sleep_stages(blocks, stages, sampling_rate=fs, epoch_duration=30)
        
        self.blocked_epochs()

    def blocked_epochs(self, ):
        assigned_stages = np.array(self.assigned_stages)
        ## wake
        self.wake_idx = np.where(assigned_stages == 0)[0]
        
        ## light sleep
        self.light_idx = np.where((assigned_stages == 1) | (assigned_stages == 2))[0]
        ## deep sleep
        self.deep_idx = np.where(assigned_stages == 3)[0]
        ## REM sleep
        self.rem_idx = np.where(assigned_stages == 5)[0]

    def calculate_wake(self):
        if len(self.wake_idx) == 0:
            return 0, 0, 0, 0, 0, 0, 0
        else:
            current = self.blocks[self.wake_idx, :]

            current_results = {
                'area':[], 'duration':[], 'fall_height':[], 'EFR':[], 
                'recover_height':[], 'ERR':[], 'EBS':[]
            }

            for i in range(current.shape[0]):

                start, nadir, end = current[i, 0], current[i, 1], current[i, 2]
                slope_idx, slope = slope_draw(sig=self.data, f0=start, f1=end)
                duration = (end - start) / self.fs
                fall_height = self.data[start] - self.data[nadir]
                EFR = fall_height / ((nadir - start) / self.fs)
                recover_height = self.data[end] - self.data[nadir]
                ERR = recover_height / ((end - nadir) / self.fs)
                whole_changes = self.data[end] - self.data[start]
                EBS = whole_changes / ((end - start) / self.fs)
                area = trapezoid(y=slope[0: -1] - self.data[start: end], dx=(1/self.fs))
                current_results['area'].append(area)
                current_results['duration'].append(duration)
                current_results['fall_height'].append(fall_height)
                current_results['EFR'].append(EFR)
                current_results['recover_height'].append(recover_height)
                current_results['ERR'].append(ERR)
                current_results['EBS'].append(EBS)

            current_table = pd.DataFrame(current_results).to_numpy()
            current_time =  len(self.wake_idx) * 30 / 60  # in minutes
            n_current_events = current_table.shape[0]
            current_means = current_table.mean(axis=0)
            current_weights = n_current_events / current_time  # events per minute
            current_features_weighted_means = current_means * current_weights
            
            self.wake_features = current_features_weighted_means

            return self.wake_features

    def calculate_light(self):
        
        if len(self.light_idx) == 0:
            return 0, 0, 0, 0, 0, 0, 0
        else:
            current = self.blocks[self.light_idx, :]

            current_results = {
                'area':[], 'duration':[], 'fall_height':[], 'EFR':[], 
                'recover_height':[], 'ERR':[], 'EBS':[]
            }

            for i in range(current.shape[0]):

                start, nadir, end = current[i, 0], current[i, 1], current[i, 2]
                slope_idx, slope = slope_draw(sig=self.data, f0=start, f1=end)
                duration = (end - start) / self.fs
                fall_height = self.data[start] - self.data[nadir]
                EFR = fall_height / ((nadir - start) / self.fs)
                recover_height = self.data[end] - self.data[nadir]
                ERR = recover_height / ((end - nadir) / self.fs)
                whole_changes = self.data[end] - self.data[start]
                EBS = whole_changes / ((end - start) / self.fs)
                area = trapezoid(y=slope[0: -1] - self.data[start: end], dx=(1/self.fs))
                current_results['area'].append(area)
                current_results['duration'].append(duration)
                current_results['fall_height'].append(fall_height)
                current_results['EFR'].append(EFR)
                current_results['recover_height'].append(recover_height)
                current_results['ERR'].append(ERR)
                current_results['EBS'].append(EBS)

            current_table = pd.DataFrame(current_results).to_numpy()
            current_time =  len(self.light_idx) * 30 / 60  # in minutes
            n_current_events = current_table.shape[0]
            current_means = current_table.mean(axis=0)
            current_weights = n_current_events / current_time  # events per minute
            current_features_weighted_means = current_means * current_weights
            
            self.light_features = current_features_weighted_means
            return self.light_features
        
    def calculate_deep(self):
        if len(self.deep_idx) == 0:
            return 0, 0, 0, 0, 0, 0, 0
        else:
            current = self.blocks[self.deep_idx, :]

            current_results = {
                'area':[], 'duration':[], 'fall_height':[], 'EFR':[], 
                'recover_height':[], 'ERR':[], 'EBS':[]
            }

            for i in range(current.shape[0]):

                start, nadir, end = current[i, 0], current[i, 1], current[i, 2]
                slope_idx, slope = slope_draw(sig=self.data, f0=start, f1=end)
                duration = (end - start) / self.fs
                fall_height = self.data[start] - self.data[nadir]
                EFR = fall_height / ((nadir - start) / self.fs)
                recover_height = self.data[end] - self.data[nadir]
                ERR = recover_height / ((end - nadir) / self.fs)
                whole_changes = self.data[end] - self.data[start]
                EBS = whole_changes / ((end - start) / self.fs)
                area = trapezoid(y=slope[0: -1] - self.data[start: end], dx=(1/self.fs))
                current_results['area'].append(area)
                current_results['duration'].append(duration)
                current_results['fall_height'].append(fall_height)
                current_results['EFR'].append(EFR)
                current_results['recover_height'].append(recover_height)
                current_results['ERR'].append(ERR)
                current_results['EBS'].append(EBS)

            current_table = pd.DataFrame(current_results).to_numpy()
            current_time =  len(self.deep_idx) * 30 / 60  # in minutes
            n_current_events = current_table.shape[0]
            current_means = current_table.mean(axis=0)
            current_weights = n_current_events / current_time  # events per minute
            current_features_weighted_means = current_means * current_weights
            
            self.deep_features = current_features_weighted_means
            return self.deep_features
        
    def calculate_rem(self):
        if len(self.rem_idx) == 0:
            return 0, 0, 0, 0, 0, 0, 0
        else:
            current = self.blocks[self.rem_idx, :]

            current_results = {
                'area':[], 'duration':[], 'fall_height':[], 'EFR':[], 
                'recover_height':[], 'ERR':[], 'EBS':[]
            }

            for i in range(current.shape[0]):

                start, nadir, end = current[i, 0], current[i, 1], current[i, 2]
                slope_idx, slope = slope_draw(sig=self.data, f0=start, f1=end)
                duration = (end - start) / self.fs
                fall_height = self.data[start] - self.data[nadir]
                EFR = fall_height / ((nadir - start) / self.fs)
                recover_height = self.data[end] - self.data[nadir]
                ERR = recover_height / ((end - nadir) / self.fs)
                whole_changes = self.data[end] - self.data[start]
                EBS = whole_changes / ((end - start) / self.fs)
                area = trapezoid(y=slope[0: -1] - self.data[start: end], dx=(1/self.fs))
                current_results['area'].append(area)
                current_results['duration'].append(duration)
                current_results['fall_height'].append(fall_height)
                current_results['EFR'].append(EFR)
                current_results['recover_height'].append(recover_height)
                current_results['ERR'].append(ERR)
                current_results['EBS'].append(EBS)

            current_table = pd.DataFrame(current_results).to_numpy()
            current_time =  len(self.rem_idx) * 30 / 60  # in minutes
            n_current_events = current_table.shape[0]
            current_means = current_table.mean(axis=0)
            current_weights = n_current_events / current_time  # events per minute
            current_features_weighted_means = current_means * current_weights
            
            self.rem_features = current_features_weighted_means
            return self.rem_features


class EventFeatures():
    """
    Calculate event-based features of the event blocks.
    data shape: (n, 3)
    return: area, duration, fall_height, EFR, recover_height, ERR, EBS, calculated_odi
    """

    def __init__(self, data, blocks, fs, stages):
        self.data = data
        self.fs = fs
        self.blocks = blocks
        self.stages = stages

        self.table = {'area':[], 'duration':[], 'fall_height':[], 'EFR':[], 'recover_height':[], 'ERR':[], 'EBS':[]}
        self.calculate()

    def mean_of_table(self):

        "orignal method: average of each col of the table"
        "New method: give a weight to each event according to sleep time"
        table_df = pd.DataFrame(self.table)

        ## count the total sleep time without wake
        sleep_time = np.sum(self.stages != 0) / 2  # numbers of sleep time with 30s epochs, to transfer to minutes, divide 2

        ## count the total number of events

        n = table_df.shape[0]

        ## mean of table
        means = table_df.mean(axis=0)
        weights = n / sleep_time  # events per minute
        weighted_means = means * weights
        odi = weights * 60  # events per hour
        return weighted_means, odi, n

    def calculate(self, ):
        """
        Calculate the seven features of the event blocks.
        """

        for i in range(self.blocks.shape[0]):

            start, nadir, end = self.blocks[i, 0], self.blocks[i, 1], self.blocks[i, 2]
            slope_idx, slope = slope_draw(sig=self.data, f0=start, f1=end)
            duration = (end - start) / self.fs
            fall_height = self.data[start] - self.data[nadir]
            EFR = fall_height / ((nadir - start) / self.fs)
            recover_height = self.data[end] - self.data[nadir]
            ERR = recover_height / ((end - nadir) / self.fs)
            whole_changes = self.data[end] - self.data[start]
            EBS = whole_changes / ((end - start) / self.fs)
            area = trapezoid(y=slope[0: -1] - self.data[start: end], dx=(1/self.fs))

            self.table['area'].append(area)
            self.table['duration'].append(duration)
            self.table['fall_height'].append(fall_height)
            self.table['EFR'].append(EFR)
            self.table['recover_height'].append(recover_height)
            self.table['ERR'].append(ERR)
            self.table['EBS'].append(EBS)

        self.table = pd.DataFrame(self.table).to_numpy()


def main__(data_frame, pack_dir_):


    res = {'nsrrid':[],
            'Wake_area':[], 'Wake_duration':[], 'Wake_fall_height':[], 'Wake_EFR':[],
            'Wake_recover_height':[], 'Wake_ERR':[], 'Wake_EBS':[],
            'Light_area':[], 'Light_duration':[], 'Light_fall_height':[], 'Light_EFR':[],
            'Light_recover_height':[], 'Light_ERR':[], 'Light_EBS':[],
            'Deep_area':[], 'Deep_duration':[], 'Deep_fall_height':[], 'Deep_EFR':[],
            'Deep_recover_height':[], 'Deep_ERR':[], 'Deep_EBS':[],
            'REM_area':[], 'REM_duration':[], 'REM_fall_height':[], 'REM_EFR':[],
            'REM_recover_height':[], 'REM_ERR':[], 'REM_EBS':[],
            'level':[]
            }


    for i in tqdm(range(data_frame.shape[0])):

        nsrrid = data_frame.loc[i, 'nsrrid']
        level = data_frame.loc[i, 'level']
        res['nsrrid'].append(nsrrid)
        res['level'].append(level)
        

        try:
            pack_path = os.path.join(pack_dir_, 'shhs1-'+str(nsrrid) + '.npy')
            pack_data = np.load(pack_path, allow_pickle=True).item()
            data_spo2 = pack_data['data']
            label_stages = pack_data['label']
            sampling_rate = pack_data['fs']

            data_spo2 = fill_nan_with_threshold(data_spo2, threshold=25)

            ## step 6.4: get the event feature

            blocks = events_main(data_spo2, fs=sampling_rate)
            events_processing = EventFeaturesStages(data=data_spo2, blocks=blocks, fs=sampling_rate, stages=label_stages)
            wake_features = events_processing.calculate_wake()
            light_features = events_processing.calculate_light()
            deep_features = events_processing.calculate_deep()
            rem_features = events_processing.calculate_rem()

            res['Wake_area'].append(wake_features[0])
            res['Wake_duration'].append(wake_features[1])
            res['Wake_fall_height'].append(wake_features[2])
            res['Wake_EFR'].append(wake_features[3])
            res['Wake_recover_height'].append(wake_features[4])
            res['Wake_ERR'].append(wake_features[5])
            res['Wake_EBS'].append(wake_features[6])

            res['Light_area'].append(light_features[0])
            res['Light_duration'].append(light_features[1])
            res['Light_fall_height'].append(light_features[2])
            res['Light_EFR'].append(light_features[3])
            res['Light_recover_height'].append(light_features[4])
            res['Light_ERR'].append(light_features[5])
            res['Light_EBS'].append(light_features[6])

            res['Deep_area'].append(deep_features[0])
            res['Deep_duration'].append(deep_features[1])
            res['Deep_fall_height'].append(deep_features[2])
            res['Deep_EFR'].append(deep_features[3])
            res['Deep_recover_height'].append(deep_features[4])
            res['Deep_ERR'].append(deep_features[5])
            res['Deep_EBS'].append(deep_features[6])

            res['REM_area'].append(rem_features[0])
            res['REM_duration'].append(rem_features[1])
            res['REM_fall_height'].append(rem_features[2])
            res['REM_EFR'].append(rem_features[3])
            res['REM_recover_height'].append(rem_features[4])
            res['REM_ERR'].append(rem_features[5])
            res['REM_EBS'].append(rem_features[6])

        except:
            print('do not find the pack file: ', nsrrid)
            res['Wake_area'].append(np.nan)
            res['Wake_duration'].append(np.nan)
            res['Wake_fall_height'].append(np.nan)
            res['Wake_EFR'].append(np.nan)
            res['Wake_recover_height'].append(np.nan)
            res['Wake_ERR'].append(np.nan)
            res['Wake_EBS'].append(np.nan)
            res['Light_area'].append(np.nan)
            res['Light_duration'].append(np.nan)
            res['Light_fall_height'].append(np.nan)
            res['Light_EFR'].append(np.nan)
            res['Light_recover_height'].append(np.nan)
            res['Light_ERR'].append(np.nan)
            res['Light_EBS'].append(np.nan)
            res['Deep_area'].append(np.nan)
            res['Deep_duration'].append(np.nan)
            res['Deep_fall_height'].append(np.nan)
            res['Deep_EFR'].append(np.nan)
            res['Deep_recover_height'].append(np.nan)
            res['Deep_ERR'].append(np.nan)
            res['Deep_EBS'].append(np.nan)
            res['REM_area'].append(np.nan)
            res['REM_duration'].append(np.nan)
            res['REM_fall_height'].append(np.nan)
            res['REM_EFR'].append(np.nan)
            res['REM_recover_height'].append(np.nan)
            res['REM_ERR'].append(np.nan)
            res['REM_EBS'].append(np.nan)
           
    res = pd.DataFrame(res)
    
    return res

if __name__ == '__main__':

    frame_original = pd.read_csv('./original_frame.csv')
    pack_dir = './small_shhs1_spo2_pack/all'

    features_df = main__(data_frame=frame_original, pack_dir_=pack_dir)
    features_df.to_csv('./saved/features_event_splited.csv', index=False)

