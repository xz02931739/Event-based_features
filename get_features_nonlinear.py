import numpy as np
import pandas as pd
import os, glob
from scipy.integrate import trapezoid

from tqdm import tqdm
from utils.cal_nonlinear import Nonlinear_tool
from utils.balanced_clinical_tools import fill_nan_with_threshold



def main__(data_frame, pack_dir_):


    res = {'nsrrid':[],
            'apen':[], 'sample_en':[], 'lz_complexity':[],
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

            ## get the non-linear feature
            apen = Nonlinear_tool(data_spo2).apen()
            sample_en = Nonlinear_tool(data_spo2).sample_en()
            lz_complexity = Nonlinear_tool(data_spo2).lz_complexity()

            res['apen'].append(apen)
            res['sample_en'].append(sample_en)
            res['lz_complexity'].append(lz_complexity)
        except:
            print('do not find the pack file: ', nsrrid)

            res['apen'].append(np.nan)
            res['sample_en'].append(np.nan)
            res['lz_complexity'].append(np.nan)

            continue
            
    res = pd.DataFrame(res)
    
    return res

if __name__ == '__main__':

    frame_original = pd.read_csv('./original_frame.csv')
    pack_dir = './small_shhs1_spo2_pack/all'

    features_df = main__(data_frame=frame_original, pack_dir_=pack_dir)
    features_df.to_csv('./saved/features_nonlinear.csv', index=False)

