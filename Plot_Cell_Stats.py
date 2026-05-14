import numpy as np
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu

from get_data_functions import *
from assemble_data_functions import *
from plot_dist_functions import *
from stat_tests import *

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

for df in arpc2ko_cells:
   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)

pooled_arpc2ko_df = pd.concat(arpc2ko_cells, ignore_index=True)

for df in wt_cells:
   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)

pooled_wt_df = pd.concat(wt_cells, ignore_index=True)


save_path = './figures/celltrack_stats'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

unique_dates_wt = pooled_wt_df.groupby('experiment')['track_name'].nunique()
unique_dates_arpc2ko = pooled_arpc2ko_df.groupby('experiment')['track_name'].nunique()

#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = unique_dates_wt.index
counts = unique_dates_wt.values
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width)
ax.bar_label(p, label_type='center')
ax.set_title('Number of WT cells from each experiment')
plt.savefig('{}/WT_numcellsbydate.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = unique_dates_arpc2ko.index
counts = unique_dates_arpc2ko.values
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width)
ax.bar_label(p, label_type='center')
ax.set_title('Number of Arpc2 KO cells from each experiment')
plt.savefig('{}/ARPC2KO_numcellsbydate.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

dt = 5 #min
duration_wt = []
duration_arpc2ko = []

for name in tracksgeo_dict.keys():
    if 'ARPC2KO' in name and '20250130' not in name and 'Bleb' not in name:
       length = len(tracksgeo_dict[name])
       duration_arpc2ko.append(length * dt)
    elif 'WT' in name and '20250130' not in name and 'Bleb' not in name:
       length = len(tracksgeo_dict[name])
       duration_wt.append(length * dt)

plt.hist(duration_wt)
plt.xlabel('Track length (min)')
plt.ylabel('Count')
plt.title('Track lengths for WT cells')
plt.savefig('{}/WT_tracklengths.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

plt.hist(duration_arpc2ko)
plt.xlabel('Track length (min)')
plt.ylabel('Count')
plt.title('Track lengths for Arpc2 KO cells')
plt.savefig('{}/ARPC2KO_tracklengths.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()