import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu

from get_data_functions import *
from assemble_data_functions import *
from plot_dist_functions import *

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 # Also set for PostScript exports
mpl.rcParams['font.family'] = 'arial'

save_path = './figures'
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

#import data 
data_dir = './data'
dipole_dict = np.load(data_dir + '/dipole_dict.npy', allow_pickle=True).item()
ellipse_dict = np.load(data_dir + '/ellipse_dict.npy', allow_pickle=True).item()
quad_dict = np.load(data_dir + '/quad_dict.npy', allow_pickle=True).item()

tracksgeo_path = data_dir + '/tracks_geo_region.pkl'
tracksgeo_dict = load_tracksgeo(tracksgeo_path, dipole_dict)

protrusion_df = pd.read_pickle(data_dir + "/protrusion.pkl")

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, protrusion_df)

numeric_pooled_arpc2ko_df = pooled_arpc2ko_df.select_dtypes(include=np.number)
numeric_pooled_wt_df = pooled_wt_df.select_dtypes(include=np.number)

save_path = './figures/DataStats'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

for df in arpc2ko_cells:
   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)

pooled_arpc2ko_df = pd.concat(arpc2ko_cells, ignore_index=True)

for df in wt_cells:
   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)

pooled_wt_df = pd.concat(wt_cells, ignore_index=True)

####### WT #########

#clears out sentinel file if it exists
open('{}/wt_stats.txt'.format(save_path),'w').close()
#create new sentinel file to write to
wt_stats = open('{}/wt_stats.txt'.format(save_path),'w')
file_lines_wt = []

params = ['area','avg_trac_mag','eccentricity','solidity','dip_ratio','turning_angle','step_length']

for var in params:
    min = np.min(pooled_wt_df[var])
    max = np.max(pooled_wt_df[var])

    file_lines_wt.append('{} - min: {} max: {}'.format(var, min, max) + '\n')

file_lines_wt.append('Number of cells: {}'.format(len(wt_cells)))

#write lines to text file 
wt_stats.writelines(file_lines_wt)
wt_stats.close() 

####### ARPC2KO #########

#clears out sentinel file if it exists
open('{}/arpc2ko_stats.txt'.format(save_path),'w').close()
#create new sentinel file to write to
arpc2ko_stats = open('{}/arpc2ko_stats.txt'.format(save_path),'w')
file_lines_arpc2ko = []

params = ['area','avg_trac_mag','eccentricity','solidity','dip_ratio','turning_angle','step_length']

for var in params:
    min = np.min(pooled_arpc2ko_df[var])
    max = np.max(pooled_arpc2ko_df[var])

    file_lines_arpc2ko.append('{} - min: {} max: {}'.format(var, min, max) + '\n')

file_lines_arpc2ko.append('Number of cells: {}'.format(len(arpc2ko_cells)))

#write lines to text file 
arpc2ko_stats.writelines(file_lines_arpc2ko)
arpc2ko_stats.close()