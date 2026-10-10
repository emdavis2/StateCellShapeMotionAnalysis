import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import pickle

pixel_size = .108 #um
pixel_area = pixel_size**2

dir1 = '/Volumes/EMD_Drive/faas_results_minA20'
dir2 = '/Volumes/EMD_Drive/faas_results_minA20crop'

save_path = '/Volumes/EMD_Drive'

fa_df = {}

cell_folders = [f"{dir1+'/'}{file}" for file in os.listdir(dir1)] + [f"{dir2+'/'}{file}" for file in os.listdir(dir2)]

for fold in cell_folders:
    exp_fold = os.listdir(fold)[0]
    raw_data_path = fold + '/' + exp_fold + '/individual_pictures/00001/raw_data'
    faai_path = fold + '/' + exp_fold + '/adhesion_props/FAAI/per_image_FAAI.csv'
    if not os.path.isdir(raw_data_path):
        print(f"{fold}: no raw_data folder")
        df = pd.DataFrame({'fa_avg_area': [np.nan], 'fa_med_area': [np.nan], 'num_fa': [np.nan], 'faai': [np.nan]})
        fa_df[fold.split('/')[-1]] = df
        continue

    area_fa_csv_path = raw_data_path + '/Area.csv'
    area_fa_csv = pd.read_csv(area_fa_csv_path,header=None)

    fa_area = area_fa_csv[0].to_numpy()

    num_fa = len(fa_area)

    if not os.path.isfile(faai_path):
        print(f"{fold}: no FAAI file")
        faai = np.nan
    else:
        faai = pd.read_csv(faai_path, header=None).iloc[0, 0]

    df = pd.DataFrame({'fa_avg_area': [np.average(fa_area) * pixel_area], 'fa_med_area': [np.median(fa_area) * pixel_area], 'num_fa': [num_fa], 'faai': [faai]})

    fa_df[fold.split('/')[-1]] = df

with open(save_path+'/fa_df.pkl', 'wb') as handle:
  pickle.dump(fa_df, handle, protocol=2)