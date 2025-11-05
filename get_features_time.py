import numpy as np
import pandas as pd
import os, glob

from tqdm import tqdm
from utils.cal_timedomain import TimeDomain_tool_with_stages
from utils.balanced_clinical_tools import fill_nan_with_threshold



"""
In this script, we extract features from the SpO2 signal and save them into a CSV file.
The features include:
- Time-domain features with sleep stages

the output CSV file include features and level will be used for classification tasks.
"""



def main__(data_frame, pack_dir_):


    res = {'nsrrid':[],
           'CT90':[], 'CT88':[], 'CT86':[],
           'Variance_SpO2':[], 'Skewness_SpO2':[], 'Kurtosis_SpO2':[],
            'Wake_mean':[], 'Wake_min':[], 'Wake_max':[],
            'Light_mean':[], 'Light_min':[], 'Light_max':[],
            'Deep_mean':[], 'Deep_min':[], 'Deep_max':[],
            'REM_mean':[], 'REM_min':[], 'REM_max':[],
            'Mean':[], 'Minimum':[], 'Maximum':[],
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

            class_timedomain = TimeDomain_tool_with_stages(x=data_spo2, fs=sampling_rate, stages=label_stages)
            all_cumulative_time_90 = class_timedomain.all_cumulative_time(thershold=90, fs=sampling_rate)
            all_cumulative_time_88 = class_timedomain.all_cumulative_time(thershold=88, fs=sampling_rate)
            all_cumulative_time_86 = class_timedomain.all_cumulative_time(thershold=86, fs=sampling_rate)

            all_variance = class_timedomain.all_variance()
            all_skewness = class_timedomain.all_skewness()
            all_kurtosis = class_timedomain.all_kurtosis()

            wake_mean, wake_min, wake_max = TimeDomain_tool_with_stages(x=fill_nan_with_threshold(data_spo2, 25), fs=sampling_rate, stages=label_stages).parameters_with_phases('wake')
            light_mean, light_min, light_max = TimeDomain_tool_with_stages(x=fill_nan_with_threshold(data_spo2, 25), fs=sampling_rate, stages=label_stages).parameters_with_phases('light')
            deep_mean, deep_min, deep_max = TimeDomain_tool_with_stages(x=fill_nan_with_threshold(data_spo2, 25), fs=sampling_rate, stages=label_stages).parameters_with_phases('deep')
            rem_mean, rem_min, rem_max = TimeDomain_tool_with_stages(x=fill_nan_with_threshold(data_spo2, 25), fs=sampling_rate, stages=label_stages).parameters_with_phases('rem')
            all_mean, all_min, all_max = TimeDomain_tool_with_stages(x=fill_nan_with_threshold(data_spo2, 25), fs=sampling_rate, stages=label_stages).parameters_with_phases('all')
            

            res['CT90'].append(all_cumulative_time_90)
            res['CT88'].append(all_cumulative_time_88)
            res['CT86'].append(all_cumulative_time_86)
            res['Variance_SpO2'].append(all_variance)
            res['Skewness_SpO2'].append(all_skewness)
            res['Kurtosis_SpO2'].append(all_kurtosis)

            res['Wake_mean'].append(wake_mean)
            res['Wake_min'].append(wake_min)
            res['Wake_max'].append(wake_max)
            res['Light_mean'].append(light_mean)
            res['Light_min'].append(light_min)
            res['Light_max'].append(light_max)
            res['Deep_mean'].append(deep_mean)
            res['Deep_min'].append(deep_min)
            res['Deep_max'].append(deep_max)
            res['REM_mean'].append(rem_mean)
            res['REM_min'].append(rem_min)
            res['REM_max'].append(rem_max)
            res['Mean'].append(all_mean)
            res['Minimum'].append(all_min)
            res['Maximum'].append(all_max)

            
        except:
            print('do not find the pack file: ', nsrrid)
            res['CT90'].append(np.nan)
            res['CT88'].append(np.nan)
            res['CT86'].append(np.nan)
            res['Variance_SpO2'].append(np.nan)
            res['Skewness_SpO2'].append(np.nan)
            res['Kurtosis_SpO2'].append(np.nan)

            res['Wake_mean'].append(np.nan)
            res['Wake_min'].append(np.nan)
            res['Wake_max'].append(np.nan)

            res['Light_mean'].append(np.nan)
            res['Light_min'].append(np.nan)
            res['Light_max'].append(np.nan)

            res['Deep_mean'].append(np.nan)
            res['Deep_min'].append(np.nan)
            res['Deep_max'].append(np.nan)

            res['REM_mean'].append(np.nan)
            res['REM_min'].append(np.nan)
            res['REM_max'].append(np.nan)

            res['Mean'].append(np.nan)
            res['Minimum'].append(np.nan)
            res['Maximum'].append(np.nan)

            continue
            
    res = pd.DataFrame(res)
    
    return res

if __name__ == '__main__':

    frame_original = pd.read_csv('./original_frame.csv')
    pack_dir = './small_shhs1_spo2_pack/all'

    features_df = main__(data_frame=frame_original, pack_dir_=pack_dir)
    features_df.to_csv('./saved/features_time.csv', index=False)
    