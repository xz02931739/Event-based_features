import numpy as np
import pandas as pd
import os, glob


from tqdm import tqdm

from utils.cal_fre_domain import FrequencyDomain_tool
from utils.balanced_clinical_tools import fill_nan_with_threshold


def main__(data_frame, pack_dir_):


    res = {'nsrrid':[],
           'PSD':[], 'Peak_fs':[], 'Spectralentropy_PSD': [],
           'PSD_mean':[], 'PSD_Variance':[], 'PSD_Skewness':[], 'PSD_Kurtosis':[],
           'level':[], }


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

            ## step 6.2: get the frequency domain feature
            power, peaks_fs = FrequencyDomain_tool(data_spo2, fs=sampling_rate).band_power([0.01, 0.03])
            spectralentropy = FrequencyDomain_tool(data_spo2, fs=sampling_rate).spectralentropy()
            mean, variance, skewness, kurtosis = FrequencyDomain_tool(data_spo2, fs=sampling_rate).statistical_moments_of_psd()


            res['PSD'].append(power)
            res['Spectralentropy_PSD'].append(spectralentropy)
            res['Peak_fs'].append(peaks_fs)
            res['PSD_mean'].append(mean)
            res['PSD_Variance'].append(variance)
            res['PSD_Skewness'].append(skewness)
            res['PSD_Kurtosis'].append(kurtosis)

        except:
            print('do not find the pack file: ', nsrrid)

            res['PSD'].append(np.nan)
            res['Peak_fs'].append(np.nan)
            res['Spectralentropy_PSD'].append(np.nan)
            res['PSD_mean'].append(np.nan)
            res['PSD_Variance'].append(np.nan)
            res['PSD_Skewness'].append(np.nan)
            res['PSD_Kurtosis'].append(np.nan)

            continue
            
    res = pd.DataFrame(res)
    
    return res

if __name__ == '__main__':

    frame_original = pd.read_csv('./original_frame.csv')
    pack_dir = './small_shhs1_spo2_pack/all'

    features_df = main__(data_frame=frame_original, pack_dir_=pack_dir)
    features_df.to_csv('./saved/features_frequency.csv', index=False)
