import numpy as np
import pandas as pd
import os, glob
from scipy.integrate import trapezoid

from tqdm import tqdm

from utils.event_instances import events_main
from utils.balanced_clinical_tools import fill_nan_with_threshold, slope_draw
from utils.exclude_nonstage import delete_nonstage

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
            'Area':[], 'Duration':[], 'Fall_height':[], 'EFR':[], 'Rise_height':[], 'ERR':[], 'EBS':[],
            'Calculated_ODI':[], 'Total_number_events':[],
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
            events_processing = EventFeatures(data=data_spo2, blocks=blocks, fs=sampling_rate, stages=label_stages)
            features, calculated_odi, total_number_events = events_processing.mean_of_table()

            res['Area'].append(features[0])
            res['Duration'].append(features[1])
            res['Fall_height'].append(features[2])
            res['EFR'].append(features[3])
            res['Rise_height'].append(features[4])
            res['ERR'].append(features[5])
            res['EBS'].append(features[6])
            res['Calculated_ODI'].append(calculated_odi)
            res['Total_number_events'].append(total_number_events)
        except:
            print('do not find the pack file: ', nsrrid)

            res['Area'].append(np.nan)
            res['Duration'].append(np.nan)
            res['Fall_height'].append(np.nan)
            res['EFR'].append(np.nan)
            res['Rise_height'].append(np.nan)
            res['ERR'].append(np.nan)
            res['EBS'].append(np.nan)
            res['Calculated_ODI'].append(np.nan)
            res['Total_number_events'].append(np.nan)

            continue
            
    res = pd.DataFrame(res)
    
    return res

if __name__ == '__main__':

    frame_original = pd.read_csv('./original_frame.csv')
    pack_dir = './small_shhs1_spo2_pack/all'

    features_df = main__(data_frame=frame_original, pack_dir_=pack_dir)
    features_df.to_csv('./saved/features_event.csv', index=False)

